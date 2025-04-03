/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: optimize.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &zp_offset, ::mlir::Operation::operand_range &input, ::mlir::DenseI64ArrayAttr &broadcast_dims_1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::chlo::BroadcastAddOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::chlo::BroadcastAddOp type";
    });
  }
  input = castedOp1.getODSOperands(0);
  zp_offset = castedOp1.getODSOperands(1);
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::DenseI64ArrayAttr>("broadcast_dimensions");(void)tblgen_attr;
    broadcast_dims_1 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_1(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &zp_offset, ::mlir::Operation::operand_range &input, ::mlir::DenseI64ArrayAttr &broadcast_dims_1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::chlo::BroadcastSubOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::chlo::BroadcastSubOp type";
    });
  }
  input = castedOp1.getODSOperands(0);
  zp_offset = castedOp1.getODSOperands(1);
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::DenseI64ArrayAttr>("broadcast_dimensions");(void)tblgen_attr;
    broadcast_dims_1 = tblgen_attr;
  }
  return ::mlir::success();
}

/* Generated from:
    tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/optimize.td:38
*/
struct optimizeConsecutiveConvCHLO_BroadcastAddOpCHLO_BroadcastAddOp : public ::mlir::RewritePattern {
  optimizeConsecutiveConvCHLO_BroadcastAddOpCHLO_BroadcastAddOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.broadcast_add", 2, context, {"chlo.broadcast_add"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range zp_offset(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_1;
    ::mlir::Operation::operand_range bias(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::BroadcastAddOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, zp_offset, input, broadcast_dims_1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    bias = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::DenseI64ArrayAttr>("broadcast_dimensions");(void)tblgen_attr;
      broadcast_dims_2 = tblgen_attr;
    }
    if (!((!broadcast_dims_1))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_1' failed to satisfy constraint: ''";
      });
    }
    if (!((!broadcast_dims_2))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_2' failed to satisfy constraint: ''";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*input.begin()).getType())))) && ([](::mlir::Type elementType) { return (::llvm::isa<::mlir::IntegerType>(elementType)); }(::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'tensor of integer values'";
      });
    }
    if (!((((*input.begin()).getDefiningOp<mhlo::ConvolutionOp>())) || (((*input.begin()).getDefiningOp<mhlo::DotGeneralOp>())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::chlo::BroadcastAddOp tblgen_BroadcastAddOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*zp_offset.begin());
      ::mlir::Value tblgen_value_1 = (*bias.begin());
      tblgen_BroadcastAddOp_0 = rewriter.create<::mlir::chlo::BroadcastAddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        /*broadcast_dimensions=*/broadcast_dims_2
      );
    }
    ::mlir::chlo::BroadcastAddOp tblgen_BroadcastAddOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_BroadcastAddOp_0.getODSResults(0).begin()));
      if (auto tmpAttr = broadcast_dims_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("broadcast_dimensions"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastAddOp_1 = rewriter.create<::mlir::chlo::BroadcastAddOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastAddOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/optimize.td:38
*/
struct optimizeConsecutiveConvCHLO_BroadcastAddOpCHLO_BroadcastSubOp : public ::mlir::RewritePattern {
  optimizeConsecutiveConvCHLO_BroadcastAddOpCHLO_BroadcastSubOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.broadcast_subtract", 2, context, {"chlo.broadcast_add", "chlo.broadcast_subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range zp_offset(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_1;
    ::mlir::Operation::operand_range bias(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::BroadcastSubOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, zp_offset, input, broadcast_dims_1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    bias = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::DenseI64ArrayAttr>("broadcast_dimensions");(void)tblgen_attr;
      broadcast_dims_2 = tblgen_attr;
    }
    if (!((!broadcast_dims_1))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_1' failed to satisfy constraint: ''";
      });
    }
    if (!((!broadcast_dims_2))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_2' failed to satisfy constraint: ''";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*input.begin()).getType())))) && ([](::mlir::Type elementType) { return (::llvm::isa<::mlir::IntegerType>(elementType)); }(::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'tensor of integer values'";
      });
    }
    if (!((((*input.begin()).getDefiningOp<mhlo::ConvolutionOp>())) || (((*input.begin()).getDefiningOp<mhlo::DotGeneralOp>())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::chlo::BroadcastSubOp tblgen_BroadcastSubOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*zp_offset.begin());
      ::mlir::Value tblgen_value_1 = (*bias.begin());
      tblgen_BroadcastSubOp_0 = rewriter.create<::mlir::chlo::BroadcastSubOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        /*broadcast_dimensions=*/broadcast_dims_2
      );
    }
    ::mlir::chlo::BroadcastAddOp tblgen_BroadcastAddOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_BroadcastSubOp_0.getODSResults(0).begin()));
      if (auto tmpAttr = broadcast_dims_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("broadcast_dimensions"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastAddOp_1 = rewriter.create<::mlir::chlo::BroadcastAddOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastAddOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/optimize.td:38
*/
struct optimizeConsecutiveConvCHLO_BroadcastSubOpCHLO_BroadcastAddOp : public ::mlir::RewritePattern {
  optimizeConsecutiveConvCHLO_BroadcastSubOpCHLO_BroadcastAddOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.broadcast_add", 2, context, {"chlo.broadcast_subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range zp_offset(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_1;
    ::mlir::Operation::operand_range bias(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::BroadcastAddOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, zp_offset, input, broadcast_dims_1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    bias = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::DenseI64ArrayAttr>("broadcast_dimensions");(void)tblgen_attr;
      broadcast_dims_2 = tblgen_attr;
    }
    if (!((!broadcast_dims_1))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_1' failed to satisfy constraint: ''";
      });
    }
    if (!((!broadcast_dims_2))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_2' failed to satisfy constraint: ''";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*input.begin()).getType())))) && ([](::mlir::Type elementType) { return (::llvm::isa<::mlir::IntegerType>(elementType)); }(::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'tensor of integer values'";
      });
    }
    if (!((((*input.begin()).getDefiningOp<mhlo::ConvolutionOp>())) || (((*input.begin()).getDefiningOp<mhlo::DotGeneralOp>())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::chlo::BroadcastSubOp tblgen_BroadcastSubOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*zp_offset.begin());
      ::mlir::Value tblgen_value_1 = (*bias.begin());
      tblgen_BroadcastSubOp_0 = rewriter.create<::mlir::chlo::BroadcastSubOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        /*broadcast_dimensions=*/broadcast_dims_2
      );
    }
    ::mlir::chlo::BroadcastSubOp tblgen_BroadcastSubOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_BroadcastSubOp_0.getODSResults(0).begin()));
      if (auto tmpAttr = broadcast_dims_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("broadcast_dimensions"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastSubOp_1 = rewriter.create<::mlir::chlo::BroadcastSubOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastSubOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/optimize.td:38
*/
struct optimizeConsecutiveConvCHLO_BroadcastSubOpCHLO_BroadcastSubOp : public ::mlir::RewritePattern {
  optimizeConsecutiveConvCHLO_BroadcastSubOpCHLO_BroadcastSubOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("chlo.broadcast_subtract", 2, context, {"chlo.broadcast_add", "chlo.broadcast_subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range zp_offset(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_1;
    ::mlir::Operation::operand_range bias(op0->getOperands());
    ::mlir::DenseI64ArrayAttr broadcast_dims_2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::chlo::BroadcastSubOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, zp_offset, input, broadcast_dims_1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    bias = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::DenseI64ArrayAttr>("broadcast_dimensions");(void)tblgen_attr;
      broadcast_dims_2 = tblgen_attr;
    }
    if (!((!broadcast_dims_1))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_1' failed to satisfy constraint: ''";
      });
    }
    if (!((!broadcast_dims_2))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'broadcast_dims_2' failed to satisfy constraint: ''";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*input.begin()).getType())))) && ([](::mlir::Type elementType) { return (::llvm::isa<::mlir::IntegerType>(elementType)); }(::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'tensor of integer values'";
      });
    }
    if (!((((*input.begin()).getDefiningOp<mhlo::ConvolutionOp>())) || (((*input.begin()).getDefiningOp<mhlo::DotGeneralOp>())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::chlo::BroadcastAddOp tblgen_BroadcastAddOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*zp_offset.begin());
      ::mlir::Value tblgen_value_1 = (*bias.begin());
      tblgen_BroadcastAddOp_0 = rewriter.create<::mlir::chlo::BroadcastAddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        /*broadcast_dimensions=*/broadcast_dims_2
      );
    }
    ::mlir::chlo::BroadcastSubOp tblgen_BroadcastSubOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_BroadcastAddOp_0.getODSResults(0).begin()));
      if (auto tmpAttr = broadcast_dims_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("broadcast_dimensions"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastSubOp_1 = rewriter.create<::mlir::chlo::BroadcastSubOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastSubOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<optimizeConsecutiveConvCHLO_BroadcastAddOpCHLO_BroadcastAddOp>(patterns.getContext());
  patterns.add<optimizeConsecutiveConvCHLO_BroadcastAddOpCHLO_BroadcastSubOp>(patterns.getContext());
  patterns.add<optimizeConsecutiveConvCHLO_BroadcastSubOpCHLO_BroadcastAddOp>(patterns.getContext());
  patterns.add<optimizeConsecutiveConvCHLO_BroadcastSubOpCHLO_BroadcastSubOp>(patterns.getContext());
}
