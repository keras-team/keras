/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: StablehloCreateCompatibilityExpanderPatterns.td                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_StablehloCreateCompatibilityExpanderPatterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!((!isa<ComplexType>(cast<ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": Non-complex element type";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_StablehloCreateCompatibilityExpanderPatterns2(
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
    external/stablehlo/stablehlo/transforms/StablehloCreateCompatibilityExpanderPatterns.td:32
*/
struct TanOp_CompatiblityExpander : public ::mlir::RewritePattern {
  TanOp_CompatiblityExpander(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("stablehlo.tan", 1, context, {"stablehlo.cosine", "stablehlo.divide", "stablehlo.sine"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::stablehlo::TanOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_StablehloCreateCompatibilityExpanderPatterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'stablehlo.tan' failed to satisfy constraint: 'Non-complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::SineOp tblgen_SineOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_SineOp_0 = rewriter.create<::mlir::stablehlo::SineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::CosineOp tblgen_CosineOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_CosineOp_1 = rewriter.create<::mlir::stablehlo::CosineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SineOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_CosineOp_1.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DivOp_2 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/StablehloCreateCompatibilityExpanderPatterns.td:40
*/
struct TanOp_ComplexElementType_CompatiblityExpander : public ::mlir::RewritePattern {
  TanOp_ComplexElementType_CompatiblityExpander(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("stablehlo.tan", 1, context, {"stablehlo.complex", "stablehlo.divide", "stablehlo.imag", "stablehlo.multiply", "stablehlo.negate", "stablehlo.real", "stablehlo.tan", "stablehlo.tanh"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::stablehlo::TanOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_StablehloCreateCompatibilityExpanderPatterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'stablehlo.tan' failed to satisfy constraint: 'Complex element type'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::stablehlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::stablehlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::TanOp tan;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      tan = rewriter.create<::mlir::stablehlo::TanOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ImagOp tblgen_ImagOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_ImagOp_1 = rewriter.create<::mlir::stablehlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::TanhOp tanh;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_1.getODSResults(0).begin());
      tanh = rewriter.create<::mlir::stablehlo::TanhOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tan.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tanh.getODSResults(0).begin());
      tblgen_ComplexOp_2 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_3 = ::mlir::stablehlo::getConstantLike(rewriter, odsLoc, 1.0, (*tan.getODSResults(0).begin())); (void)nativeVar_3;
    ::mlir::stablehlo::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*tan.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tanh.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::stablehlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::NegOp tblgen_NegOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_4.getODSResults(0).begin());
      tblgen_NegOp_5 = rewriter.create<::mlir::stablehlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::stablehlo::ComplexOp tblgen_ComplexOp_6;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_3;
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_5.getODSResults(0).begin());
      tblgen_ComplexOp_6 = rewriter.create<::mlir::stablehlo::ComplexOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::stablehlo::DivOp tblgen_DivOp_7;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ComplexOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ComplexOp_6.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DivOp_7 = rewriter.create<::mlir::stablehlo::DivOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivOp_7.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<TanOp_CompatiblityExpander>(patterns.getContext());
  patterns.add<TanOp_ComplexElementType_CompatiblityExpander>(patterns.getContext());
}
