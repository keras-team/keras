/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ChloDecompositionPatterns.td                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ChloDecompositionPatterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!((!isa<ComplexType>(cast<ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": Non-complex element type";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ChloDecompositionPatterns2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!((isa<ComplexType>(cast<ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": Complex element type";
    });
  }
  return ::mlir::success();
}
/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:51
*/
struct GeneratedConvert0 : public ::mlir::RewritePattern {
  GeneratedConvert0(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.atan", 1, context, {"stablehlo.atan2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AtanOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.atan' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*input.begin())); (void)nativeVar_0;
    ::mlir::stablehlo::Atan2Op tblgen_Atan2Op_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back(nativeVar_0);
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Atan2Op_1 = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_Atan2Op_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:60
*/
struct GeneratedConvert1 : public ::mlir::RewritePattern {
  GeneratedConvert1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.atanh", 1, context, {"stablehlo.abs", "stablehlo.compare", "stablehlo.log_plus_one", "stablehlo.multiply", "stablehlo.negate", "stablehlo.select", "stablehlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AtanhOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.atanh' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::AbsOp tblgen_AbsOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_AbsOp_0 = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_1 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*input.begin())); (void)nativeVar_1;
    auto nativeVar_2 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_2;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AbsOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_1;
      tblgen_CompareOp_3 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GT),
        /*compare_type=*/nativeVar_2
      );
    }
    auto nativeVar_4 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, NAN, (*input.begin())); (void)nativeVar_4;
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_Log1pOp_5 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_NegOp_6 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_NegOp_6.getODSResults(0).begin());
      tblgen_Log1pOp_7 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SubtractOp tblgen_SubtractOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_Log1pOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_Log1pOp_7.getODSResults(0).begin());
      tblgen_SubtractOp_8 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_9 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0.5, (*input.begin())); (void)nativeVar_9;
    ::mlir::stablehlo::MulOp tblgen_MulOp_10;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubtractOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_9;
      tblgen_MulOp_10 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_11;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_CompareOp_3.getODSResults(0).begin()));
      tblgen_values.push_back(nativeVar_4);
      tblgen_values.push_back((*tblgen_MulOp_10.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectOp_11 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectOp_11.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:82
*/
struct GeneratedConvert2 : public ::mlir::RewritePattern {
  GeneratedConvert2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.conj", 1, context, {"stablehlo.complex", "stablehlo.imag", "stablehlo.negate", "stablehlo.real"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range v(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::ConjOp>(op0); (void)castedOp0;
    v = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*v.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp tblgen_ImagOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*v.begin());
      tblgen_ImagOp_1 = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_1.getODSResults(0).begin());
      tblgen_NegOp_2 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_RealOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_NegOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_3 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:87
*/
struct GeneratedConvert3 : public ::mlir::RewritePattern {
  GeneratedConvert3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.is_inf", 1, context, {"chlo.is_pos_inf", "stablehlo.abs"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::IsInfOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.is_inf' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::AbsOp tblgen_AbsOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_AbsOp_0 = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::chlo::IsPosInfOp tblgen_IsPosInfOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AbsOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IsPosInfOp_1 = rewriter.create<::mlir::chlo::IsPosInfOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IsPosInfOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:94
*/
struct GeneratedConvert4 : public ::mlir::RewritePattern {
  GeneratedConvert4(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.is_pos_inf", 1, context, {"stablehlo.compare"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::IsPosInfOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.is_pos_inf' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::stablehlo::getConstantLikeInfValue(rewriter, odsLoc, (*input.begin()), /*negative=*/false); (void)nativeVar_0;
    auto nativeVar_1 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_1;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back(nativeVar_0);
      if (auto tmpAttr = ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("comparison_direction"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("compare_type"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CompareOp_2 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CompareOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:104
*/
struct GeneratedConvert5 : public ::mlir::RewritePattern {
  GeneratedConvert5(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.is_neg_inf", 1, context, {"stablehlo.compare"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::IsNegInfOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.is_neg_inf' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::stablehlo::getConstantLikeInfValue(rewriter, odsLoc, (*input.begin()), /*negative=*/true); (void)nativeVar_0;
    auto nativeVar_1 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_1;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back(nativeVar_0);
      if (auto tmpAttr = ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("comparison_direction"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("compare_type"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CompareOp_2 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CompareOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:113
*/
struct GeneratedConvert6 : public ::mlir::RewritePattern {
  GeneratedConvert6(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.tan", 1, context, {"stablehlo.tan"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::TanOp>(op0); (void)castedOp0;
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::TanOp tblgen_TanOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_TanOp_0 = rewriter.create<::mlir::stablehlo::TanOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_TanOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td:116
*/
struct GeneratedConvert7 : public ::mlir::RewritePattern {
  GeneratedConvert7(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.constant", 1, context, {"stablehlo.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::ElementsAttr v;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::ConstantOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::ElementsAttr>("value");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'chlo.constant' to have attribute 'value' of type '::mlir::ElementsAttr'";
        });
      }
      v = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::ConstantOp tblgen_ConstantOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = v) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("value"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ConstantOp_0 = rewriter.create<::mlir::stablehlo::ConstantOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ConstantOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:43
*/
struct GeneratedConvert8 : public ::mlir::RewritePattern {
  GeneratedConvert8(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo._asin_acos_kernel", 1, context, {"stablehlo.abs", "stablehlo.add", "stablehlo.and", "stablehlo.compare", "stablehlo.complex", "stablehlo.divide", "stablehlo.imag", "stablehlo.log", "stablehlo.log_plus_one", "stablehlo.maximum", "stablehlo.minimum", "stablehlo.multiply", "stablehlo.negate", "stablehlo.not", "stablehlo.real", "stablehlo.select", "stablehlo.sqrt", "stablehlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AsinAcosKernelOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo._asin_acos_kernel' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::RealOp signed_x;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_x = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AbsOp x;
    {
      ::mlir::Value tblgen_value_0 = (*signed_x.getODSResults(0).begin());
      x = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp signed_y;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_y = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AbsOp y;
    {
      ::mlir::Value tblgen_value_0 = (*signed_y.getODSResults(0).begin());
      y = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MaxOp tblgen_MaxOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      tblgen_MaxOp_0 = rewriter.create<::mlir::stablehlo::MaxOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_1 = ::mlir::stablehlo::getConstantLikeMaxFiniteValue(rewriter, odsLoc, (*signed_x.getODSResults(0).begin())); (void)nativeVar_1;
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_2;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_1;
      tblgen_SqrtOp_2 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_3 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 8, (*signed_x.getODSResults(0).begin())); (void)nativeVar_3;
    ::mlir::stablehlo::DivOp safe_max;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SqrtOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_3;
      safe_max = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_4 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_4;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MaxOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*safe_max.getODSResults(0).begin());
      tblgen_CompareOp_5 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_4
      );
    }
    auto one = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*signed_x.getODSResults(0).begin())); (void)one;
    auto nativeVar_6 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_6;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      tblgen_CompareOp_7 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LE),
        /*compare_type=*/nativeVar_6
      );
    }
    auto half = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0.5, (*signed_x.getODSResults(0).begin())); (void)half;
    ::mlir::stablehlo::AddOp xp1;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      xp1 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AbsOp abs_xp1;
    {
      ::mlir::Value tblgen_value_0 = (*xp1.getODSResults(0).begin());
      abs_xp1 = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MaxOp _hypot_1_mx;
    {
      ::mlir::Value tblgen_value_0 = (*abs_xp1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      _hypot_1_mx = rewriter.create<::mlir::stablehlo::MaxOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MinOp mn;
    {
      ::mlir::Value tblgen_value_0 = (*abs_xp1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      mn = rewriter.create<::mlir::stablehlo::MinOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_8 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_8;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_1_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*mn.getODSResults(0).begin());
      tblgen_CompareOp_9 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_8
      );
    }
    auto two = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 2, (*signed_x.getODSResults(0).begin())); (void)two;
    ::mlir::stablehlo::SqrtOp sqrt_two;
    {
      ::mlir::Value tblgen_value_0 = two;
      sqrt_two = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_10;
    {
      ::mlir::Value tblgen_value_0 = (*sqrt_two.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_1_mx.getODSResults(0).begin());
      tblgen_MulOp_10 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp mn_over_mx;
    {
      ::mlir::Value tblgen_value_0 = (*mn.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_1_mx.getODSResults(0).begin());
      mn_over_mx = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp _hypot_1_r;
    {
      ::mlir::Value tblgen_value_0 = (*mn_over_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*mn_over_mx.getODSResults(0).begin());
      _hypot_1_r = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_11;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*_hypot_1_r.getODSResults(0).begin());
      tblgen_AddOp_11 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp sqa;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_11.getODSResults(0).begin());
      sqa = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_12 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_12;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_13;
    {
      ::mlir::Value tblgen_value_0 = (*sqa.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      tblgen_CompareOp_13 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_12
      );
    }
    auto zero = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, (*signed_x.getODSResults(0).begin())); (void)zero;
    auto nativeVar_14 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_14;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_15;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_1_r.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = zero;
      tblgen_CompareOp_15 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GT),
        /*compare_type=*/nativeVar_14
      );
    }
    ::mlir::stablehlo::AndOp tblgen_AndOp_16;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_13.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_CompareOp_15.getODSResults(0).begin());
      tblgen_AndOp_16 = rewriter.create<::mlir::stablehlo::AndOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_17;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_1_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_1_r.getODSResults(0).begin());
      tblgen_MulOp_17 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_18;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_17.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = two;
      tblgen_DivOp_18 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_19;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_1_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_18.getODSResults(0).begin());
      tblgen_AddOp_19 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_20;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_1_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*sqa.getODSResults(0).begin());
      tblgen_MulOp_20 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_21;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AndOp_16.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_19.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_MulOp_20.getODSResults(0).begin());
      tblgen_SelectOp_21 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp r;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_9.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_10.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_21.getODSResults(0).begin());
      r = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SubtractOp xm1;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      xm1 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AbsOp abs_xm1;
    {
      ::mlir::Value tblgen_value_0 = (*xm1.getODSResults(0).begin());
      abs_xm1 = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MaxOp _hypot_2_mx;
    {
      ::mlir::Value tblgen_value_0 = (*abs_xm1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      _hypot_2_mx = rewriter.create<::mlir::stablehlo::MaxOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MinOp _hypot_2_mn;
    {
      ::mlir::Value tblgen_value_0 = (*abs_xm1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      _hypot_2_mn = rewriter.create<::mlir::stablehlo::MinOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_22 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_22;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_23;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_2_mn.getODSResults(0).begin());
      tblgen_CompareOp_23 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_22
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_24;
    {
      ::mlir::Value tblgen_value_0 = (*sqrt_two.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_2_mx.getODSResults(0).begin());
      tblgen_MulOp_24 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp _hypot_2_mn_over_mx;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_mn.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_2_mx.getODSResults(0).begin());
      _hypot_2_mn_over_mx = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp _hypot_2_r;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_mn_over_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_2_mn_over_mx.getODSResults(0).begin());
      _hypot_2_r = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_25;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*_hypot_2_r.getODSResults(0).begin());
      tblgen_AddOp_25 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp _hypot_2_sqa;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_25.getODSResults(0).begin());
      _hypot_2_sqa = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_26 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_26;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_27;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_sqa.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      tblgen_CompareOp_27 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_26
      );
    }
    auto nativeVar_28 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_28;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_29;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_r.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = zero;
      tblgen_CompareOp_29 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GT),
        /*compare_type=*/nativeVar_28
      );
    }
    ::mlir::stablehlo::AndOp tblgen_AndOp_30;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_27.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_CompareOp_29.getODSResults(0).begin());
      tblgen_AndOp_30 = rewriter.create<::mlir::stablehlo::AndOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_31;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_2_r.getODSResults(0).begin());
      tblgen_MulOp_31 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_32;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_31.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = two;
      tblgen_DivOp_32 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_33;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_32.getODSResults(0).begin());
      tblgen_AddOp_33 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_34;
    {
      ::mlir::Value tblgen_value_0 = (*_hypot_2_mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*_hypot_2_sqa.getODSResults(0).begin());
      tblgen_MulOp_34 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_35;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AndOp_30.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_33.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_MulOp_34.getODSResults(0).begin());
      tblgen_SelectOp_35 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp s;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_23.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_24.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_35.getODSResults(0).begin());
      s = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_36;
    {
      ::mlir::Value tblgen_value_0 = (*r.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*s.getODSResults(0).begin());
      tblgen_AddOp_36 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp a;
    {
      ::mlir::Value tblgen_value_0 = half;
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_36.getODSResults(0).begin());
      a = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_37;
    {
      ::mlir::Value tblgen_value_0 = (*a.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*x.getODSResults(0).begin());
      tblgen_AddOp_37 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp half_apx;
    {
      ::mlir::Value tblgen_value_0 = half;
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_37.getODSResults(0).begin());
      half_apx = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp yy;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      yy = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp rpxp1;
    {
      ::mlir::Value tblgen_value_0 = (*r.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*xp1.getODSResults(0).begin());
      rpxp1 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_38;
    {
      ::mlir::Value tblgen_value_0 = (*yy.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rpxp1.getODSResults(0).begin());
      tblgen_DivOp_38 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SubtractOp smxm1;
    {
      ::mlir::Value tblgen_value_0 = (*s.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*xm1.getODSResults(0).begin());
      smxm1 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_39;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_DivOp_38.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*smxm1.getODSResults(0).begin());
      tblgen_AddOp_39 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_40;
    {
      ::mlir::Value tblgen_value_0 = (*half_apx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_39.getODSResults(0).begin());
      tblgen_MulOp_40 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_41;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_40.getODSResults(0).begin());
      tblgen_SqrtOp_41 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_42;
    {
      ::mlir::Value tblgen_value_0 = (*half_apx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rpxp1.getODSResults(0).begin());
      tblgen_DivOp_42 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp spxm1;
    {
      ::mlir::Value tblgen_value_0 = (*s.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*xm1.getODSResults(0).begin());
      spxm1 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_43;
    {
      ::mlir::Value tblgen_value_0 = (*half_apx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*spxm1.getODSResults(0).begin());
      tblgen_DivOp_43 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_44;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_DivOp_42.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_43.getODSResults(0).begin());
      tblgen_AddOp_44 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_45;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_44.getODSResults(0).begin());
      tblgen_SqrtOp_45 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_46;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SqrtOp_45.getODSResults(0).begin());
      tblgen_MulOp_46 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_47;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_7.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SqrtOp_41.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_MulOp_46.getODSResults(0).begin());
      tblgen_SelectOp_47 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_48;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_47.getODSResults(0).begin());
      tblgen_SelectOp_48 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    auto nativeVar_49 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1000000000000.0, (*signed_x.getODSResults(0).begin())); (void)nativeVar_49;
    ::mlir::stablehlo::MulOp tblgen_MulOp_50;
    {
      ::mlir::Value tblgen_value_0 = (*safe_max.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_49;
      tblgen_MulOp_50 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_51 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_51;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_52;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_50.getODSResults(0).begin());
      tblgen_CompareOp_52 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_51
      );
    }
    auto nativeVar_53 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1e-06, (*signed_x.getODSResults(0).begin())); (void)nativeVar_53;
    ::mlir::stablehlo::MulOp tblgen_MulOp_54;
    {
      ::mlir::Value tblgen_value_0 = (*safe_max.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_53;
      tblgen_MulOp_54 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_55 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 100.0, (*signed_x.getODSResults(0).begin())); (void)nativeVar_55;
    ::mlir::stablehlo::MulOp tblgen_MulOp_56;
    {
      ::mlir::Value tblgen_value_0 = (*safe_max.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_55;
      tblgen_MulOp_56 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp safe_max_opt;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_52.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_54.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_MulOp_56.getODSResults(0).begin());
      safe_max_opt = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    auto nativeVar_57 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_57;
    ::mlir::stablehlo::CompareOp y_gt_safe_max_opt;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*safe_max_opt.getODSResults(0).begin());
      y_gt_safe_max_opt = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_57
      );
    }
    ::mlir::stablehlo::SelectOp mx;
    {
      ::mlir::Value tblgen_value_0 = (*y_gt_safe_max_opt.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*x.getODSResults(0).begin());
      mx = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_58;
    {
      ::mlir::Value tblgen_value_0 = (*y_gt_safe_max_opt.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*safe_max_opt.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*safe_max.getODSResults(0).begin());
      tblgen_SelectOp_58 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    auto nativeVar_59 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_59;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_60;
    {
      ::mlir::Value tblgen_value_0 = (*mx.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SelectOp_58.getODSResults(0).begin());
      tblgen_CompareOp_60 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_59
      );
    }
    ::mlir::stablehlo::LogOp tblgen_LogOp_61;
    {
      ::mlir::Value tblgen_value_0 = two;
      tblgen_LogOp_61 = rewriter.create<::mlir::stablehlo::LogOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::LogOp tblgen_LogOp_62;
    {
      ::mlir::Value tblgen_value_0 = (*mx.getODSResults(0).begin());
      tblgen_LogOp_62 = rewriter.create<::mlir::stablehlo::LogOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_63;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LogOp_61.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_LogOp_62.getODSResults(0).begin());
      tblgen_AddOp_63 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_64 = ::mlir::stablehlo::getConstantLikeInfValue(rewriter, odsLoc, (*signed_y.getODSResults(0).begin()), /*negative=*/false); (void)nativeVar_64;
    auto nativeVar_65 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_65;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_66;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_64;
      tblgen_CompareOp_66 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_65
      );
    }
    ::mlir::stablehlo::NotOp tblgen_NotOp_67;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_66.getODSResults(0).begin());
      tblgen_NotOp_67 = rewriter.create<::mlir::stablehlo::NotOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AndOp tblgen_AndOp_68;
    {
      ::mlir::Value tblgen_value_0 = (*y_gt_safe_max_opt.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NotOp_67.getODSResults(0).begin());
      tblgen_AndOp_68 = rewriter.create<::mlir::stablehlo::AndOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_69;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      tblgen_DivOp_69 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp xoy;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AndOp_68.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_69.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = zero;
      xoy = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_70;
    {
      ::mlir::Value tblgen_value_0 = (*xoy.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*xoy.getODSResults(0).begin());
      tblgen_MulOp_70 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_71;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_70.getODSResults(0).begin());
      tblgen_Log1pOp_71 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_72;
    {
      ::mlir::Value tblgen_value_0 = half;
      ::mlir::Value tblgen_value_1 = (*tblgen_Log1pOp_71.getODSResults(0).begin());
      tblgen_MulOp_72 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_73;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_63.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_72.getODSResults(0).begin());
      tblgen_AddOp_73 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_74 = ::mlir::stablehlo::getConstantLikeSmallestNormalizedValue(rewriter, odsLoc, (*signed_x.getODSResults(0).begin())); (void)nativeVar_74;
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_75;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_74;
      tblgen_SqrtOp_75 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_76 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 4, (*signed_x.getODSResults(0).begin())); (void)nativeVar_76;
    ::mlir::stablehlo::MulOp tblgen_MulOp_77;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SqrtOp_75.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_76;
      tblgen_MulOp_77 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_78 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_78;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_79;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_77.getODSResults(0).begin());
      tblgen_CompareOp_79 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_78
      );
    }
    auto nativeVar_80 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_80;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_81;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      tblgen_CompareOp_81 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_80
      );
    }
    ::mlir::stablehlo::AndOp logical_and_lt_y_safe_min_lt_x_one;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_79.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_CompareOp_81.getODSResults(0).begin());
      logical_and_lt_y_safe_min_lt_x_one = rewriter.create<::mlir::stablehlo::AndOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_82;
    {
      ::mlir::Value tblgen_value_0 = (*xp1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*xm1.getODSResults(0).begin());
      tblgen_MulOp_82 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp ap1;
    {
      ::mlir::Value tblgen_value_0 = (*a.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      ap1 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_83;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_82.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*ap1.getODSResults(0).begin());
      tblgen_DivOp_83 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_84;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_DivOp_83.getODSResults(0).begin());
      tblgen_NegOp_84 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_85 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_85;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_86;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      tblgen_CompareOp_86 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_85
      );
    }
    ::mlir::stablehlo::MulOp half_yy;
    {
      ::mlir::Value tblgen_value_0 = half;
      ::mlir::Value tblgen_value_1 = (*yy.getODSResults(0).begin());
      half_yy = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp divide_half_yy_rpxp1;
    {
      ::mlir::Value tblgen_value_0 = (*half_yy.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rpxp1.getODSResults(0).begin());
      divide_half_yy_rpxp1 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_87;
    {
      ::mlir::Value tblgen_value_0 = half;
      ::mlir::Value tblgen_value_1 = (*spxm1.getODSResults(0).begin());
      tblgen_MulOp_87 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_88;
    {
      ::mlir::Value tblgen_value_0 = (*divide_half_yy_rpxp1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_87.getODSResults(0).begin());
      tblgen_AddOp_88 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_89 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1.5, (*signed_x.getODSResults(0).begin())); (void)nativeVar_89;
    auto nativeVar_90 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_90;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_91;
    {
      ::mlir::Value tblgen_value_0 = (*a.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_89;
      tblgen_CompareOp_91 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LE),
        /*compare_type=*/nativeVar_90
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_92;
    {
      ::mlir::Value tblgen_value_0 = (*half_yy.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*smxm1.getODSResults(0).begin());
      tblgen_DivOp_92 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_93;
    {
      ::mlir::Value tblgen_value_0 = (*divide_half_yy_rpxp1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_92.getODSResults(0).begin());
      tblgen_AddOp_93 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SubtractOp tblgen_SubtractOp_94;
    {
      ::mlir::Value tblgen_value_0 = (*a.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      tblgen_SubtractOp_94 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_95;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_91.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_93.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SubtractOp_94.getODSResults(0).begin());
      tblgen_SelectOp_95 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp x_ge_1_or_not;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_86.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_88.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_95.getODSResults(0).begin());
      x_ge_1_or_not = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp am1;
    {
      ::mlir::Value tblgen_value_0 = (*logical_and_lt_y_safe_min_lt_x_one.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_84.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*x_ge_1_or_not.getODSResults(0).begin());
      am1 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_96;
    {
      ::mlir::Value tblgen_value_0 = (*am1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*ap1.getODSResults(0).begin());
      tblgen_MulOp_96 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp sq;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_96.getODSResults(0).begin());
      sq = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_97;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*sq.getODSResults(0).begin());
      tblgen_DivOp_97 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_98;
    {
      ::mlir::Value tblgen_value_0 = (*am1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*sq.getODSResults(0).begin());
      tblgen_AddOp_98 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_99;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_98.getODSResults(0).begin());
      tblgen_Log1pOp_99 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_100;
    {
      ::mlir::Value tblgen_value_0 = (*logical_and_lt_y_safe_min_lt_x_one.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_97.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_Log1pOp_99.getODSResults(0).begin());
      tblgen_SelectOp_100 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_101;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_60.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_73.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_100.getODSResults(0).begin());
      tblgen_SelectOp_101 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_102;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SelectOp_48.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_SelectOp_101.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_102 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_102.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:390
*/
struct GeneratedConvert9 : public ::mlir::RewritePattern {
  GeneratedConvert9(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.asin", 1, context, {"chlo._asin_acos_kernel", "stablehlo.atan2", "stablehlo.compare", "stablehlo.complex", "stablehlo.imag", "stablehlo.negate", "stablehlo.real", "stablehlo.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AsinOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.asin' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::RealOp signed_x;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_x = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::chlo::AsinAcosKernelOp asin_acos_kernel_z;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      asin_acos_kernel_z = rewriter.create<::mlir::chlo::AsinAcosKernelOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*asin_acos_kernel_z.getODSResults(0).begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::Atan2Op real;
    {
      ::mlir::Value tblgen_value_0 = (*signed_x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_0.getODSResults(0).begin());
      real = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::ImagOp signed_y;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_y = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp imag_asin_acos_kernel_z;
    {
      ::mlir::Value tblgen_value_0 = (*asin_acos_kernel_z.getODSResults(0).begin());
      imag_asin_acos_kernel_z = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_1 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, imag_asin_acos_kernel_z); (void)nativeVar_1;
    auto nativeVar_2 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_2;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*signed_y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_1;
      tblgen_CompareOp_3 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_2
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*imag_asin_acos_kernel_z.getODSResults(0).begin());
      tblgen_NegOp_4 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*imag_asin_acos_kernel_z.getODSResults(0).begin());
      tblgen_SelectOp_5 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*real.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_SelectOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_6 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:413
*/
struct GeneratedConvert10 : public ::mlir::RewritePattern {
  GeneratedConvert10(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.asin", 1, context, {"stablehlo.add", "stablehlo.atan2", "stablehlo.multiply", "stablehlo.sqrt", "stablehlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AsinOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.asin' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto one = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*x.begin())); (void)one;
    ::mlir::stablehlo::SubtractOp tblgen_SubtractOp_0;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*x.begin());
      tblgen_SubtractOp_0 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_1;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*x.begin());
      tblgen_AddOp_1 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubtractOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_1.getODSResults(0).begin());
      tblgen_MulOp_2 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_2.getODSResults(0).begin());
      tblgen_SqrtOp_3 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_4;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*tblgen_SqrtOp_3.getODSResults(0).begin());
      tblgen_AddOp_4 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Atan2Op ta;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_4.getODSResults(0).begin());
      ta = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*ta.getODSResults(0).begin()));
      tblgen_values.push_back((*ta.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddOp_5 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddOp_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:465
*/
struct GeneratedConvert11 : public ::mlir::RewritePattern {
  GeneratedConvert11(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.acos", 1, context, {"chlo._asin_acos_kernel", "stablehlo.atan2", "stablehlo.compare", "stablehlo.complex", "stablehlo.imag", "stablehlo.negate", "stablehlo.real", "stablehlo.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AcosOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.acos' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::chlo::AsinAcosKernelOp asin_acos_kernel_z;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      asin_acos_kernel_z = rewriter.create<::mlir::chlo::AsinAcosKernelOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*asin_acos_kernel_z.getODSResults(0).begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::RealOp signed_x;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_x = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::Atan2Op tblgen_Atan2Op_1;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*signed_x.getODSResults(0).begin());
      tblgen_Atan2Op_1 = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::ImagOp signed_y;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_y = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_2 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, (*signed_y.getODSResults(0).begin())); (void)nativeVar_2;
    auto nativeVar_3 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_3;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*signed_y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_2;
      tblgen_CompareOp_4 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_3
      );
    }
    ::mlir::stablehlo::ImagOp imag_asin_acos_kernel_z;
    {
      ::mlir::Value tblgen_value_0 = (*asin_acos_kernel_z.getODSResults(0).begin());
      imag_asin_acos_kernel_z = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*imag_asin_acos_kernel_z.getODSResults(0).begin());
      tblgen_NegOp_5 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*imag_asin_acos_kernel_z.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_NegOp_5.getODSResults(0).begin());
      tblgen_SelectOp_6 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_7;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_Atan2Op_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_SelectOp_6.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_7 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_7.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:490
*/
struct GeneratedConvert12 : public ::mlir::RewritePattern {
  GeneratedConvert12(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.acos", 1, context, {"stablehlo.add", "stablehlo.atan2", "stablehlo.multiply", "stablehlo.sqrt", "stablehlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AcosOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.acos' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto constant_1 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*x.begin())); (void)constant_1;
    ::mlir::stablehlo::SubtractOp tblgen_SubtractOp_0;
    {
      ::mlir::Value tblgen_value_0 = constant_1;
      ::mlir::Value tblgen_value_1 = (*x.begin());
      tblgen_SubtractOp_0 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_1;
    {
      ::mlir::Value tblgen_value_0 = constant_1;
      ::mlir::Value tblgen_value_1 = (*x.begin());
      tblgen_AddOp_1 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubtractOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_1.getODSResults(0).begin());
      tblgen_MulOp_2 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_2.getODSResults(0).begin());
      tblgen_SqrtOp_3 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::Atan2Op tblgen_Atan2Op_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SqrtOp_3.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Atan2Op_4 = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_Atan2Op_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:527
*/
struct GeneratedConvert13 : public ::mlir::RewritePattern {
  GeneratedConvert13(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.acosh", 1, context, {"chlo._asin_acos_kernel", "stablehlo.atan2", "stablehlo.compare", "stablehlo.complex", "stablehlo.imag", "stablehlo.negate", "stablehlo.real", "stablehlo.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AcoshOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.acosh' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::chlo::AsinAcosKernelOp w;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      w = rewriter.create<::mlir::chlo::AsinAcosKernelOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp tblgen_ImagOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*w.getODSResults(0).begin());
      tblgen_ImagOp_0 = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp signed_y;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_y = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_1 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, (*signed_y.getODSResults(0).begin())); (void)nativeVar_1;
    auto nativeVar_2 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_2;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*signed_y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_1;
      tblgen_CompareOp_3 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_2
      );
    }
    ::mlir::stablehlo::RealOp tblgen_RealOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*w.getODSResults(0).begin());
      tblgen_RealOp_4 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::RealOp signed_x;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_x = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::Atan2Op imag;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*signed_x.getODSResults(0).begin());
      imag = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      tblgen_NegOp_5 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*imag.getODSResults(0).begin());
      tblgen_SelectOp_6 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_7;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ImagOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_SelectOp_6.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_7 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_7.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:560
*/
struct GeneratedConvert14 : public ::mlir::RewritePattern {
  GeneratedConvert14(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.acosh", 1, context, {"stablehlo.add", "stablehlo.compare", "stablehlo.divide", "stablehlo.log", "stablehlo.log_plus_one", "stablehlo.multiply", "stablehlo.select", "stablehlo.sqrt", "stablehlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AcoshOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.acosh' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::stablehlo::getConstantLikeMaxFiniteValue(rewriter, odsLoc, (*x.begin())); (void)nativeVar_0;
    auto constant_2 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 2, (*x.begin())); (void)constant_2;
    ::mlir::stablehlo::DivOp tblgen_DivOp_1;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_0;
      ::mlir::Value tblgen_value_1 = constant_2;
      tblgen_DivOp_1 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_2 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_2;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_1.getODSResults(0).begin());
      tblgen_CompareOp_3 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_2
      );
    }
    ::mlir::stablehlo::LogOp tblgen_LogOp_4;
    {
      ::mlir::Value tblgen_value_0 = constant_2;
      tblgen_LogOp_4 = rewriter.create<::mlir::stablehlo::LogOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::LogOp tblgen_LogOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      tblgen_LogOp_5 = rewriter.create<::mlir::stablehlo::LogOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LogOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_LogOp_5.getODSResults(0).begin());
      tblgen_AddOp_6 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto constant_1 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*x.begin())); (void)constant_1;
    ::mlir::stablehlo::SubtractOp tblgen_SubtractOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = constant_1;
      tblgen_SubtractOp_7 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp sqrt_subtract_x_constant_1;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubtractOp_7.getODSResults(0).begin());
      sqrt_subtract_x_constant_1 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = constant_1;
      tblgen_AddOp_8 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_8.getODSResults(0).begin());
      tblgen_SqrtOp_9 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_10;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SqrtOp_9.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*sqrt_subtract_x_constant_1.getODSResults(0).begin());
      tblgen_AddOp_10 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_11;
    {
      ::mlir::Value tblgen_value_0 = (*sqrt_subtract_x_constant_1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_10.getODSResults(0).begin());
      tblgen_MulOp_11 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_12;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_11.getODSResults(0).begin());
      tblgen_Log1pOp_12 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_13;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_CompareOp_3.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_AddOp_6.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_Log1pOp_12.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectOp_13 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectOp_13.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:604
*/
struct GeneratedConvert15 : public ::mlir::RewritePattern {
  GeneratedConvert15(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.asinh", 1, context, {"chlo._asin_acos_kernel", "stablehlo.atan2", "stablehlo.compare", "stablehlo.complex", "stablehlo.imag", "stablehlo.negate", "stablehlo.real", "stablehlo.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AsinhOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.asinh' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::RealOp signed_x;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_x = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_0 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, (*signed_x.getODSResults(0).begin())); (void)nativeVar_0;
    auto nativeVar_1 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_1;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*signed_x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_0;
      tblgen_CompareOp_2 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_1
      );
    }
    ::mlir::stablehlo::ImagOp signed_y;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      signed_y = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*signed_y.getODSResults(0).begin());
      tblgen_NegOp_3 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_NegOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*signed_x.getODSResults(0).begin());
      tblgen_ComplexOp_4 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::chlo::AsinAcosKernelOp w;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ComplexOp_4.getODSResults(0).begin());
      w = rewriter.create<::mlir::chlo::AsinAcosKernelOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp imag_w;
    {
      ::mlir::Value tblgen_value_0 = (*w.getODSResults(0).begin());
      imag_w = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*imag_w.getODSResults(0).begin());
      tblgen_NegOp_5 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*imag_w.getODSResults(0).begin());
      tblgen_SelectOp_6 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::RealOp tblgen_RealOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*w.getODSResults(0).begin());
      tblgen_RealOp_7 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::Atan2Op tblgen_Atan2Op_8;
    {
      ::mlir::Value tblgen_value_0 = (*signed_y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_7.getODSResults(0).begin());
      tblgen_Atan2Op_8 = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_9;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SelectOp_6.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_Atan2Op_8.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_9 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_9.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:657
*/
struct GeneratedConvert16 : public ::mlir::RewritePattern {
  GeneratedConvert16(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.asinh", 1, context, {"stablehlo.abs", "stablehlo.add", "stablehlo.compare", "stablehlo.divide", "stablehlo.log", "stablehlo.log_plus_one", "stablehlo.multiply", "stablehlo.select", "stablehlo.sign", "stablehlo.sqrt"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AsinhOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.asinh' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::SignOp tblgen_SignOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      tblgen_SignOp_0 = rewriter.create<::mlir::stablehlo::SignOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AbsOp ax;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ax = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_1 = ::mlir::stablehlo::getConstantLikeMaxFiniteValue(rewriter, odsLoc, (*x.begin())); (void)nativeVar_1;
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_2;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_1;
      tblgen_SqrtOp_2 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_3 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_3;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SqrtOp_2.getODSResults(0).begin());
      tblgen_CompareOp_4 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_3
      );
    }
    auto nativeVar_5 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 2, (*x.begin())); (void)nativeVar_5;
    ::mlir::stablehlo::LogOp tblgen_LogOp_6;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_5;
      tblgen_LogOp_6 = rewriter.create<::mlir::stablehlo::LogOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::LogOp tblgen_LogOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      tblgen_LogOp_7 = rewriter.create<::mlir::stablehlo::LogOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LogOp_6.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_LogOp_7.getODSResults(0).begin());
      tblgen_AddOp_8 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp ax2;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*ax.getODSResults(0).begin());
      ax2 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto one = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*x.begin())); (void)one;
    ::mlir::stablehlo::AddOp tblgen_AddOp_9;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*ax2.getODSResults(0).begin());
      tblgen_AddOp_9 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SqrtOp tblgen_SqrtOp_10;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_9.getODSResults(0).begin());
      tblgen_SqrtOp_10 = rewriter.create<::mlir::stablehlo::SqrtOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_11;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*tblgen_SqrtOp_10.getODSResults(0).begin());
      tblgen_AddOp_11 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_12;
    {
      ::mlir::Value tblgen_value_0 = (*ax2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_11.getODSResults(0).begin());
      tblgen_DivOp_12 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_13;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_12.getODSResults(0).begin());
      tblgen_AddOp_13 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_14;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_13.getODSResults(0).begin());
      tblgen_Log1pOp_14 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_15;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_Log1pOp_14.getODSResults(0).begin());
      tblgen_SelectOp_15 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_16;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SignOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_SelectOp_15.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_16 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulOp_16.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:685
*/
struct GeneratedConvert17 : public ::mlir::RewritePattern {
  GeneratedConvert17(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.atan", 1, context, {"chlo.atanh", "stablehlo.complex", "stablehlo.imag", "stablehlo.negate", "stablehlo.real"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AtanOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.atan' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::ImagOp tblgen_ImagOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      tblgen_ImagOp_0 = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_0.getODSResults(0).begin());
      tblgen_NegOp_1 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::RealOp tblgen_RealOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      tblgen_RealOp_2 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_NegOp_1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_2.getODSResults(0).begin());
      tblgen_ComplexOp_3 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::chlo::AtanhOp w;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ComplexOp_3.getODSResults(0).begin());
      w = rewriter.create<::mlir::chlo::AtanhOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp tblgen_ImagOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*w.getODSResults(0).begin());
      tblgen_ImagOp_4 = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::RealOp tblgen_RealOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*w.getODSResults(0).begin());
      tblgen_RealOp_5 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_5.getODSResults(0).begin());
      tblgen_NegOp_6 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_7;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ImagOp_4.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_NegOp_6.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_7 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_7.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms\ChloDecompositionPatternsMath.td:772
*/
struct GeneratedConvert18 : public ::mlir::RewritePattern {
  GeneratedConvert18(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.atanh", 1, context, {"stablehlo.abs", "stablehlo.add", "stablehlo.and", "stablehlo.atan2", "stablehlo.compare", "stablehlo.complex", "stablehlo.divide", "stablehlo.imag", "stablehlo.log_plus_one", "stablehlo.multiply", "stablehlo.or", "stablehlo.real", "stablehlo.select", "stablehlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range z(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::AtanhOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_ChloDecompositionPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'chlo.atanh' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    z = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::RealOp x;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      x = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto zero = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, (*x.getODSResults(0).begin())); (void)zero;
    auto nativeVar_0 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_0;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = zero;
      tblgen_CompareOp_1 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_0
      );
    }
    auto one = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1, (*x.getODSResults(0).begin())); (void)one;
    auto constant_neg1 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, -1, (*x.getODSResults(0).begin())); (void)constant_neg1;
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      ::mlir::Value tblgen_value_2 = constant_neg1;
      tblgen_SelectOp_2 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    auto nativeVar_3 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 4, (*x.getODSResults(0).begin())); (void)nativeVar_3;
    ::mlir::stablehlo::AbsOp ax;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ax = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto largest = ::mlir::stablehlo::getConstantLikeMaxFiniteValue(rewriter, odsLoc, (*x.getODSResults(0).begin())); (void)largest;
    auto nativeVar_4 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1e+308, (*x.getODSResults(0).begin())); (void)nativeVar_4;
    auto nativeVar_5 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_5;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_6;
    {
      ::mlir::Value tblgen_value_0 = largest;
      ::mlir::Value tblgen_value_1 = nativeVar_4;
      tblgen_CompareOp_6 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GT),
        /*compare_type=*/nativeVar_5
      );
    }
    auto nativeVar_7 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 9007199254740994.0, (*x.getODSResults(0).begin())); (void)nativeVar_7;
    auto nativeVar_8 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1e+38, (*x.getODSResults(0).begin())); (void)nativeVar_8;
    auto nativeVar_9 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_9;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_10;
    {
      ::mlir::Value tblgen_value_0 = largest;
      ::mlir::Value tblgen_value_1 = nativeVar_8;
      tblgen_CompareOp_10 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GT),
        /*compare_type=*/nativeVar_9
      );
    }
    auto nativeVar_11 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 16777218.0, (*x.getODSResults(0).begin())); (void)nativeVar_11;
    auto nativeVar_12 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 2050.0, (*x.getODSResults(0).begin())); (void)nativeVar_12;
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_13;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_10.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_11;
      ::mlir::Value tblgen_value_2 = nativeVar_12;
      tblgen_SelectOp_13 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp inv_negeps;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_6.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_7;
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_13.getODSResults(0).begin());
      inv_negeps = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::MulOp safe_max;
    {
      ::mlir::Value tblgen_value_0 = (*inv_negeps.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*inv_negeps.getODSResults(0).begin());
      safe_max = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_14 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_14;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_15;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*safe_max.getODSResults(0).begin());
      tblgen_CompareOp_15 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_14
      );
    }
    ::mlir::stablehlo::ImagOp y;
    {
      ::mlir::Value tblgen_value_0 = (*z.begin());
      y = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::AbsOp ay;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ay = rewriter.create<::mlir::stablehlo::AbsOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    auto nativeVar_16 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_16;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_17;
    {
      ::mlir::Value tblgen_value_0 = (*ay.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*safe_max.getODSResults(0).begin());
      tblgen_CompareOp_17 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_16
      );
    }
    ::mlir::stablehlo::AndOp in_safe_region;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_15.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_CompareOp_17.getODSResults(0).begin());
      in_safe_region = rewriter.create<::mlir::stablehlo::AndOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SubtractOp naxm1;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*ax.getODSResults(0).begin());
      naxm1 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_18;
    {
      ::mlir::Value tblgen_value_0 = (*naxm1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*naxm1.getODSResults(0).begin());
      tblgen_MulOp_18 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp y2;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      y2 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_19;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_18.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y2.getODSResults(0).begin());
      tblgen_AddOp_19 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_20;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_19.getODSResults(0).begin());
      tblgen_DivOp_20 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_21;
    {
      ::mlir::Value tblgen_value_0 = (*ay.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*inv_negeps.getODSResults(0).begin());
      tblgen_MulOp_21 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_22 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_22;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_23;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_21.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*ax.getODSResults(0).begin());
      tblgen_CompareOp_23 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::LT),
        /*compare_type=*/nativeVar_22
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_24;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*ax.getODSResults(0).begin());
      tblgen_DivOp_24 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto constant_posinf = ::mlir::stablehlo::getConstantLikeInfValue(rewriter, odsLoc, (*x.getODSResults(0).begin()), /*negative=*/false); (void)constant_posinf;
    auto nativeVar_25 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_25;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_26;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = constant_posinf;
      tblgen_CompareOp_26 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_25
      );
    }
    auto constant_neginf = ::mlir::stablehlo::getConstantLikeInfValue(rewriter, odsLoc, (*x.getODSResults(0).begin()), /*negative=*/true); (void)constant_neginf;
    auto nativeVar_27 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_27;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_28;
    {
      ::mlir::Value tblgen_value_0 = (*x.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = constant_neginf;
      tblgen_CompareOp_28 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_27
      );
    }
    ::mlir::stablehlo::OrOp tblgen_OrOp_29;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_26.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_CompareOp_28.getODSResults(0).begin());
      tblgen_OrOp_29 = rewriter.create<::mlir::stablehlo::OrOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_30 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_30;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_31;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = constant_posinf;
      tblgen_CompareOp_31 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_30
      );
    }
    auto nativeVar_32 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_32;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_33;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = constant_neginf;
      tblgen_CompareOp_33 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::EQ),
        /*compare_type=*/nativeVar_32
      );
    }
    ::mlir::stablehlo::OrOp tblgen_OrOp_34;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_31.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_CompareOp_33.getODSResults(0).begin());
      tblgen_OrOp_34 = rewriter.create<::mlir::stablehlo::OrOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::OrOp tblgen_OrOp_35;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_OrOp_29.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_OrOp_34.getODSResults(0).begin());
      tblgen_OrOp_35 = rewriter.create<::mlir::stablehlo::OrOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_36;
    {
      ::mlir::Value tblgen_value_0 = (*ax.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      tblgen_DivOp_36 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_37;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*ax.getODSResults(0).begin());
      tblgen_DivOp_37 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_38;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_DivOp_36.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_37.getODSResults(0).begin());
      tblgen_AddOp_38 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_39;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_38.getODSResults(0).begin());
      tblgen_DivOp_39 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_40;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_DivOp_39.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      tblgen_DivOp_40 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_41;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_OrOp_35.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = zero;
      ::mlir::Value tblgen_value_2 = (*tblgen_DivOp_40.getODSResults(0).begin());
      tblgen_SelectOp_41 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_42;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_23.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_24.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_41.getODSResults(0).begin());
      tblgen_SelectOp_42 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_43;
    {
      ::mlir::Value tblgen_value_0 = (*in_safe_region.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_20.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_SelectOp_42.getODSResults(0).begin());
      tblgen_SelectOp_43 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_44;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_3;
      ::mlir::Value tblgen_value_1 = (*tblgen_SelectOp_43.getODSResults(0).begin());
      tblgen_MulOp_44 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Log1pOp tblgen_Log1pOp_45;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_44.getODSResults(0).begin());
      tblgen_Log1pOp_45 = rewriter.create<::mlir::stablehlo::Log1pOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_46;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SelectOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_Log1pOp_45.getODSResults(0).begin());
      tblgen_MulOp_46 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_47 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0.25, (*x.getODSResults(0).begin())); (void)nativeVar_47;
    ::mlir::stablehlo::MulOp tblgen_MulOp_48;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_46.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_47;
      tblgen_MulOp_48 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_49;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y.getODSResults(0).begin());
      tblgen_AddOp_49 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::AddOp tblgen_AddOp_50;
    {
      ::mlir::Value tblgen_value_0 = one;
      ::mlir::Value tblgen_value_1 = (*ax.getODSResults(0).begin());
      tblgen_AddOp_50 = rewriter.create<::mlir::stablehlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::MulOp tblgen_MulOp_51;
    {
      ::mlir::Value tblgen_value_0 = (*naxm1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_50.getODSResults(0).begin());
      tblgen_MulOp_51 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SubtractOp tblgen_SubtractOp_52;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_51.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*y2.getODSResults(0).begin());
      tblgen_SubtractOp_52 = rewriter.create<::mlir::stablehlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::Atan2Op tblgen_Atan2Op_53;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddOp_49.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SubtractOp_52.getODSResults(0).begin());
      tblgen_Atan2Op_53 = rewriter.create<::mlir::stablehlo::Atan2Op>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_54 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0, (*y.getODSResults(0).begin())); (void)nativeVar_54;
    auto nativeVar_55 = ::mlir::stablehlo::ComparisonTypeAttr(); (void)nativeVar_55;
    ::mlir::stablehlo::CompareOp tblgen_CompareOp_56;
    {
      ::mlir::Value tblgen_value_0 = (*y.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_54;
      tblgen_CompareOp_56 = rewriter.create<::mlir::stablehlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::stablehlo::ComparisonDirection::GE),
        /*compare_type=*/nativeVar_55
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_57;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CompareOp_56.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = one;
      ::mlir::Value tblgen_value_2 = constant_neg1;
      tblgen_SelectOp_57 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    auto nativeVar_58 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, M_PI, (*x.getODSResults(0).begin())); (void)nativeVar_58;
    ::mlir::stablehlo::MulOp tblgen_MulOp_59;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SelectOp_57.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_58;
      tblgen_MulOp_59 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::SelectOp tblgen_SelectOp_60;
    {
      ::mlir::Value tblgen_value_0 = (*in_safe_region.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_Atan2Op_53.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_MulOp_59.getODSResults(0).begin());
      tblgen_SelectOp_60 = rewriter.create<::mlir::stablehlo::SelectOp>(odsLoc,
        /*pred=*/tblgen_value_0,
        /*on_true=*/tblgen_value_1,
        /*on_false=*/tblgen_value_2
      );
    }
    auto nativeVar_61 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 0.5, (*x.getODSResults(0).begin())); (void)nativeVar_61;
    ::mlir::stablehlo::MulOp tblgen_MulOp_62;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SelectOp_60.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_61;
      tblgen_MulOp_62 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_63;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_48.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_62.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_63 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_63.getODSResults(0) }) {
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
  patterns.add<GeneratedConvert15>(patterns.getContext());
  patterns.add<GeneratedConvert16>(patterns.getContext());
  patterns.add<GeneratedConvert17>(patterns.getContext());
  patterns.add<GeneratedConvert18>(patterns.getContext());
}
