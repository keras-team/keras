/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArithCanonicalization.td                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((::llvm::isa<::mlir::IntegerAttr>(attr)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": arbitrary integer attribute";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((true))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": any attribute";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization3(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute 0";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization4(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getIntegerAttr(rewriter.getIntegerType(1), 1)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute 1";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization5(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((::llvm::isa<::mlir::TypedAttr>(attr)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": TypedAttr instance";
    });
  }
  return ::mlir::success();
}
static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Attribute &c0) {
  (void)tblgen_ops;
    ::mlir::Attribute arg1_0;
    if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0)))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0))) return ::mlir::failure";
      });
    }
    c0 = arg1_0;
    if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization1(rewriter, op0, arg1_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'arbitrary integer attribute'"))) {
      return ::mlir::failure();
    }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_1(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x, ::mlir::Attribute &c0, ::mlir::arith::IntegerOverflowFlagsAttr &ovf1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::AddIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  {
    auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
    if (!(op2)){
      return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
        diag << "There's no operation that defines operand 1 of castedOp1";
      });
    }
    if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
      return ::mlir::failure();
    }
    tblgen_ops.push_back(op2);
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
    ovf1 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_2(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Attribute &c1) {
  (void)tblgen_ops;
    ::mlir::Attribute arg1_0;
    if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0)))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0))) return ::mlir::failure";
      });
    }
    c1 = arg1_0;
    if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization1(rewriter, op0, arg1_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'arbitrary integer attribute'"))) {
      return ::mlir::failure();
    }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_3(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Attribute &c0) {
  (void)tblgen_ops;
    ::mlir::Attribute arg1_0;
    if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0)))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0))) return ::mlir::failure";
      });
    }
    c0 = arg1_0;
    if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization2(rewriter, op0, arg1_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'any attribute'"))) {
      return ::mlir::failure();
    }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_4(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Attribute &c0, ::mlir::Operation::operand_range &x, ::mlir::arith::IntegerOverflowFlagsAttr &ovf1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::SubIOp type";
    });
  }
  {
    auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
    if (!(op2)){
      return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
        diag << "There's no operation that defines operand 0 of castedOp1";
      });
    }
    if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
      return ::mlir::failure();
    }
    tblgen_ops.push_back(op2);
  }
  x = castedOp1.getODSOperands(1);
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
    ovf1 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_5(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x, ::mlir::Attribute &c0, ::mlir::arith::IntegerOverflowFlagsAttr &ovf1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::SubIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  {
    auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
    if (!(op2)){
      return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
        diag << "There's no operation that defines operand 1 of castedOp1";
      });
    }
    if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
      return ::mlir::failure();
    }
    tblgen_ops.push_back(op2);
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
    ovf1 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_6(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_7(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &y) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
    });
  }
  y = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_8(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_9(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &y) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
    });
  }
  y = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_10(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::NegFOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::NegFOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_11(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &y) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::NegFOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::NegFOp type";
    });
  }
  y = castedOp1.getODSOperands(0);
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_12(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops) {
  (void)tblgen_ops;
    ::mlir::Attribute arg1_0;
    if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0)))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "::mlir::success(::mlir::matchPattern(op0->getResult(0), ::mlir::m_Constant(&arg1_0))) return ::mlir::failure";
      });
    }
    if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization4(rewriter, op0, arg1_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'constant attribute 1'"))) {
      return ::mlir::failure();
    }
  return ::mlir::success();
}

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:43
*/
struct AddIAddConstant : public ::mlir::RewritePattern {
  AddIAddConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 2, context, {"arith.addi", "arith.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::AddIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, x, c0, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::AddIOp tblgen_AddIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_AddIOp_3 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:80
*/
struct AddIMulNegativeOneLhs : public ::mlir::RewritePattern {
  AddIMulNegativeOneLhs(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 2, context, {"arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::MulIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::MulIOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_3(rewriter, op2, tblgen_ops, c0))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
        ovf1 = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    y = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }
    if (!(((succeeded(getIntOrSplatIntValue(c0)))) && ((getIntOrSplatIntValue(c0)->isAllOnes())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'c0' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::SubIOp tblgen_SubIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*y.begin()));
      tblgen_values.push_back((*x.begin()));
      if (auto tmpAttr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_0 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:72
*/
struct AddIMulNegativeOneRhs : public ::mlir::RewritePattern {
  AddIMulNegativeOneRhs(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 2, context, {"arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::MulIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::MulIOp type";
        });
      }
      y = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_3(rewriter, op2, tblgen_ops, c0))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
        ovf1 = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }
    if (!(((succeeded(getIntOrSplatIntValue(c0)))) && ((getIntOrSplatIntValue(c0)->isAllOnes())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'c0' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::SubIOp tblgen_SubIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_0 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:59
*/
struct AddISubConstantLHS : public ::mlir::RewritePattern {
  AddISubConstantLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::AddIOp res;
    ::mlir::Attribute c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_4(rewriter, op1, tblgen_ops, c0, x, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::SubIOp tblgen_SubIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_3 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:51
*/
struct AddISubConstantRHS : public ::mlir::RewritePattern {
  AddISubConstantRHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 2, context, {"arith.addi", "arith.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::AddIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_5(rewriter, op1, tblgen_ops, x, c0, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c1, c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::AddIOp tblgen_AddIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_AddIOp_3 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:101
*/
struct AddUIExtendedToAddI : public ::mlir::RewritePattern {
  AddUIExtendedToAddI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addui_extended", 1, context, {"arith.addi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::AddUIExtendedOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddUIExtendedOp>(op0); (void)castedOp0;
    res = castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);
    if (!(((*res.getODSResults(1).begin()).getUses().empty()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res__1' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AddIOp tblgen_AddIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_AddIOp_0 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:337
*/
struct AndOfExtSI : public ::mlir::RewritePattern {
  AndOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.andi", 3, context, {"arith.andi", "arith.extsi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AndIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_6(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AndIOp tblgen_AndIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_AndIOp_0 = rewriter.create<::mlir::arith::AndIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AndIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_1 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:331
*/
struct AndOfExtUI : public ::mlir::RewritePattern {
  AndOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.andi", 3, context, {"arith.andi", "arith.extui"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AndIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_9(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AndIOp tblgen_AndIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_AndIOp_0 = rewriter.create<::mlir::arith::AndIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AndIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_1 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:315
*/
struct BitcastOfBitcast : public ::mlir::RewritePattern {
  BitcastOfBitcast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.bitcast", 2, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::BitcastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::BitcastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::BitcastOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
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
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:233
*/
struct CmpIExtSI : public ::mlir::RewritePattern {
  CmpIExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.cmpi", 3, context, {"arith.cmpi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::CmpIPredicateAttr pred;
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::CmpIOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::CmpIPredicateAttr>("predicate");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'arith.cmpi' to have attribute 'predicate' of type '::mlir::arith::CmpIPredicateAttr'";
        });
      }
      pred = tblgen_attr;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
        });
      }
      b = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*a.begin()).getType() == (*b.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'a, b' failed to satisfy constraint: ''";
      });
    }
    if (!((pred.getValue() == arith::CmpIPredicate::eq || pred.getValue() == arith::CmpIPredicate::ne))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'pred' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::CmpIOp tblgen_CmpIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = pred) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpIOp_0 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:244
*/
struct CmpIExtUI : public ::mlir::RewritePattern {
  CmpIExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.cmpi", 3, context, {"arith.cmpi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::CmpIPredicateAttr pred;
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::CmpIOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::CmpIPredicateAttr>("predicate");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'arith.cmpi' to have attribute 'predicate' of type '::mlir::arith::CmpIPredicateAttr'";
        });
      }
      pred = tblgen_attr;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
        });
      }
      b = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*a.begin()).getType() == (*b.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'a, b' failed to satisfy constraint: ''";
      });
    }
    if (!((pred.getValue() == arith::CmpIPredicate::eq || pred.getValue() == arith::CmpIPredicate::ne))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'pred' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::CmpIOp tblgen_CmpIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = pred) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpIOp_0 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:437
*/
struct DivFOfNegF : public ::mlir::RewritePattern {
  DivFOfNegF(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.divf", 3, context, {"arith.divf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::arith::FastMathFlagsAttr fmf;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::DivFOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_10(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_11(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
      fmf = tblgen_attr;
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::DivFOp tblgen_DivFOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = fmf) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_DivFOp_0 = rewriter.create<::mlir::arith::DivFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivFOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:323
*/
struct ExtSIOfExtUI : public ::mlir::RewritePattern {
  ExtSIOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.extsi", 2, context, {"arith.extui"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_0 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:292
*/
struct IndexCastOfExtSI : public ::mlir::RewritePattern {
  IndexCastOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_cast", 2, context, {"arith.index_cast"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_6(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::IndexCastOp tblgen_IndexCastOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IndexCastOp_0 = rewriter.create<::mlir::arith::IndexCastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IndexCastOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:286
*/
struct IndexCastOfIndexCast : public ::mlir::RewritePattern {
  IndexCastOfIndexCast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_cast", 2, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::IndexCastOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::IndexCastOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*res.getODSResults(0).begin()).getType() == (*x.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, x' failed to satisfy constraint: ''";
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
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:306
*/
struct IndexCastUIOfExtUI : public ::mlir::RewritePattern {
  IndexCastUIOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_castui", 2, context, {"arith.index_castui"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::IndexCastUIOp tblgen_IndexCastUIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IndexCastUIOp_0 = rewriter.create<::mlir::arith::IndexCastUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IndexCastUIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:300
*/
struct IndexCastUIOfIndexCastUI : public ::mlir::RewritePattern {
  IndexCastUIOfIndexCastUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_castui", 2, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::IndexCastUIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::IndexCastUIOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*res.getODSResults(0).begin()).getType() == (*x.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, x' failed to satisfy constraint: ''";
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
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:426
*/
struct MulFOfNegF : public ::mlir::RewritePattern {
  MulFOfNegF(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.mulf", 3, context, {"arith.mulf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::arith::FastMathFlagsAttr fmf;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::MulFOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_10(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_11(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
      fmf = tblgen_attr;
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::MulFOp tblgen_MulFOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = fmf) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_MulFOp_0 = rewriter.create<::mlir::arith::MulFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulFOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:88
*/
struct MulIMulIConstant : public ::mlir::RewritePattern {
  MulIMulIConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.muli", 2, context, {"arith.constant", "arith.muli"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::MulIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::MulIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::MulIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::MulIOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
        ovf1 = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = mulIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::MulIOp tblgen_MulIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_MulIOp_3 = rewriter.create<::mlir::arith::MulIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:183
*/
struct MulSIExtendedRHSOne : public ::mlir::RewritePattern {
  MulSIExtendedRHSOne(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.mulsi_extended", 1, context, {"arith.cmpi", "arith.constant", "arith.extsi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c1;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::MulSIExtendedOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
        ::mlir::Attribute arg1_0;
        if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op1->getResult(0), ::mlir::m_Constant(&arg1_0)))))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "::mlir::success(::mlir::matchPattern(op1->getResult(0), ::mlir::m_Constant(&arg1_0))) return ::mlir::failure";
          });
        }
        c1 = arg1_0;
        if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization2(rewriter, op1, arg1_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'any attribute'"))) {
          return ::mlir::failure();
        }
      tblgen_ops.push_back(op1);
    }
    if (!(((succeeded(getIntOrSplatIntValue(c1)))) && ((getIntOrSplatIntValue(c1)->isStrictlyPositive())) && ((*getIntOrSplatIntValue(c1) == 1)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'c1' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }
    auto nativeVar_0 = rewriter.getZeroAttr((*x.begin()).getType()); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::CmpIOp tblgen_CmpIOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_1.getODSResults(0).begin());
      tblgen_CmpIOp_2 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc,
        ::mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(), ::mlir::arith::CmpIPredicate::slt),
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_CmpIOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(1)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_3 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:170
*/
struct MulSIExtendedToMulI : public ::mlir::RewritePattern {
  MulSIExtendedToMulI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.mulsi_extended", 1, context, {"arith.muli"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::MulSIExtendedOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::MulSIExtendedOp>(op0); (void)castedOp0;
    res = castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);
    if (!(((*res.getODSResults(1).begin()).getUses().empty()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res__1' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::MulIOp tblgen_MulIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_MulIOp_0 = rewriter.create<::mlir::arith::MulIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:198
*/
struct MulUIExtendedToMulI : public ::mlir::RewritePattern {
  MulUIExtendedToMulI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.mului_extended", 1, context, {"arith.muli"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::MulUIExtendedOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::MulUIExtendedOp>(op0); (void)castedOp0;
    res = castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);
    if (!(((*res.getODSResults(1).begin()).getUses().empty()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res__1' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::MulIOp tblgen_MulIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_MulIOp_0 = rewriter.create<::mlir::arith::MulIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:353
*/
struct OrOfExtSI : public ::mlir::RewritePattern {
  OrOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.ori", 3, context, {"arith.extsi", "arith.ori"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::OrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_6(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::OrIOp tblgen_OrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_OrIOp_0 = rewriter.create<::mlir::arith::OrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_OrIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_1 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:347
*/
struct OrOfExtUI : public ::mlir::RewritePattern {
  OrOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.ori", 3, context, {"arith.extui", "arith.ori"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::OrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_9(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::OrIOp tblgen_OrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_OrIOp_0 = rewriter.create<::mlir::arith::OrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_OrIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_1 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:270
*/
struct RedundantSelectFalse : public ::mlir::RewritePattern {
  RedundantSelectFalse(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.select", 2, context, {"arith.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range pred(op0->getOperands());
    ::mlir::Operation::operand_range pred0(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SelectOp>(op0); (void)castedOp0;
    pred = castedOp0.getODSOperands(0);
    a = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::SelectOp type";
        });
      }
      pred0 = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      c = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*pred.begin() == *pred0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'pred' and 'pred0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::SelectOp tblgen_SelectOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*pred.begin()));
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*c.begin()));
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

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:265
*/
struct RedundantSelectTrue : public ::mlir::RewritePattern {
  RedundantSelectTrue(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.select", 2, context, {"arith.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range pred(op0->getOperands());
    ::mlir::Operation::operand_range pred0(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range c(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SelectOp>(op0); (void)castedOp0;
    pred = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::SelectOp type";
        });
      }
      pred0 = castedOp1.getODSOperands(0);
      a = castedOp1.getODSOperands(1);
      b = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    c = castedOp0.getODSOperands(2);
    if (!(*pred.begin() == *pred0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'pred' and 'pred0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::SelectOp tblgen_SelectOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*pred.begin()));
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*c.begin()));
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

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:275
*/
struct SelectI1ToNot : public ::mlir::RewritePattern {
  SelectI1ToNot(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.select", 1, context, {"arith.constant", "arith.xori"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range pred(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SelectOp>(op0); (void)castedOp0;
    pred = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
        ::mlir::Attribute arg1_0;
        if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op1->getResult(0), ::mlir::m_Constant(&arg1_0)))))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "::mlir::success(::mlir::matchPattern(op1->getResult(0), ::mlir::m_Constant(&arg1_0))) return ::mlir::failure";
          });
        }
        if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization3(rewriter, op1, arg1_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'constant attribute 0'"))) {
          return ::mlir::failure();
        }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_12(rewriter, op1, tblgen_ops))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 1)
      );
    }
    ::mlir::arith::XOrIOp tblgen_XOrIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*pred.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_0.getODSResults(0).begin()));
      tblgen_XOrIOp_1 = rewriter.create<::mlir::arith::XOrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_XOrIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:259
*/
struct SelectNotCond : public ::mlir::RewritePattern {
  SelectNotCond(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.select", 2, context, {"arith.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range pred(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Attribute ones;
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::XOrIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::XOrIOp type";
        });
      }
      pred = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
          ::mlir::Attribute arg2_0;
          if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op2->getResult(0), ::mlir::m_Constant(&arg2_0)))))){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "::mlir::success(::mlir::matchPattern(op2->getResult(0), ::mlir::m_Constant(&arg2_0))) return ::mlir::failure";
            });
          }
          ones = arg2_0;
          if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization1(rewriter, op2, arg2_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'arbitrary integer attribute'"))) {
            return ::mlir::failure();
          }
        tblgen_ops.push_back(op2);
      }
      tblgen_ops.push_back(op1);
    }
    a = castedOp0.getODSOperands(1);
    b = castedOp0.getODSOperands(2);
    if (!(((succeeded(getIntOrSplatIntValue(ones)))) && ((getIntOrSplatIntValue(ones)->isAllOnes())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'ones' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::SelectOp tblgen_SelectOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*pred.begin()));
      tblgen_values.push_back((*b.begin()));
      tblgen_values.push_back((*a.begin()));
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

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:119
*/
struct SubILHSAddConstant : public ::mlir::RewritePattern {
  SubILHSAddConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::Attribute c1;
    ::mlir::arith::SubIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, x, c0, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c1, c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::SubIOp tblgen_SubIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_3 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:151
*/
struct SubILHSSubConstantLHS : public ::mlir::RewritePattern {
  SubILHSSubConstantLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.addi", "arith.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::Attribute c1;
    ::mlir::arith::SubIOp res;
    ::mlir::Attribute c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_4(rewriter, op1, tblgen_ops, c0, x, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c1, c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::AddIOp tblgen_AddIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_AddIOp_3 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:143
*/
struct SubILHSSubConstantRHS : public ::mlir::RewritePattern {
  SubILHSSubConstantRHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::Attribute c1;
    ::mlir::arith::SubIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_5(rewriter, op1, tblgen_ops, x, c0, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::SubIOp tblgen_SubIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_3 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:111
*/
struct SubIRHSAddConstant : public ::mlir::RewritePattern {
  SubIRHSAddConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.addi", "arith.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::SubIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, x, c0, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::AddIOp tblgen_AddIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_AddIOp_2 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:135
*/
struct SubIRHSSubConstantLHS : public ::mlir::RewritePattern {
  SubIRHSSubConstantLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::SubIOp res;
    ::mlir::Attribute c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_4(rewriter, op1, tblgen_ops, c0, x, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::SubIOp tblgen_SubIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_3 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:127
*/
struct SubIRHSSubConstantRHS : public ::mlir::RewritePattern {
  SubIRHSSubConstantRHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute c1;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::SubIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_5(rewriter, op1, tblgen_ops, x, c0, ovf1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::SubIOp tblgen_SubIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_3 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:159
*/
struct SubISubILHSRHSLHS : public ::mlir::RewritePattern {
  SubISubILHSRHSLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::SubIOp res;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range x0(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::arith::IntegerOverflowFlagsAttr ovf2;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::SubIOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      y = castedOp1.getODSOperands(1);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
        ovf1 = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    x0 = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
      ovf2 = tblgen_attr;
    }
    if (!(*x.begin() == *x0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'x' and 'x0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getZeroAttr((*y.begin()).getType()); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = mergeOverflowFlags(ovf1, ovf2); (void)nativeVar_2;
    ::mlir::arith::SubIOp tblgen_SubIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_3 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:378
*/
struct TruncIExtSIToExtSI : public ::mlir::RewritePattern {
  TruncIExtSIToExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.trunci", 2, context, {"arith.extsi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::TruncIOp tr;
    ::mlir::arith::ExtSIOp ext;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::TruncIOp>(op0); (void)castedOp0;
    tr = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
        });
      }
      ext = castedOp1;
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((getScalarOrElementWidth((*ext.getODSResults(0).begin())) > getScalarOrElementWidth((*tr.getODSResults(0).begin())))) && ((getScalarOrElementWidth((*tr.getODSResults(0).begin())) > 0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'ext, tr' failed to satisfy constraint: ''";
      });
    }
    if (!(((getScalarOrElementWidth((*tr.getODSResults(0).begin())) > getScalarOrElementWidth((*x.begin())))) && ((getScalarOrElementWidth((*x.begin())) > 0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'tr, x' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_0 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:385
*/
struct TruncIExtUIToExtUI : public ::mlir::RewritePattern {
  TruncIExtUIToExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.trunci", 2, context, {"arith.extui"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::TruncIOp tr;
    ::mlir::arith::ExtUIOp ext;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::TruncIOp>(op0); (void)castedOp0;
    tr = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
        });
      }
      ext = castedOp1;
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((getScalarOrElementWidth((*ext.getODSResults(0).begin())) > getScalarOrElementWidth((*tr.getODSResults(0).begin())))) && ((getScalarOrElementWidth((*tr.getODSResults(0).begin())) > 0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'ext, tr' failed to satisfy constraint: ''";
      });
    }
    if (!(((getScalarOrElementWidth((*tr.getODSResults(0).begin())) > getScalarOrElementWidth((*x.begin())))) && ((getScalarOrElementWidth((*x.begin())) > 0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'tr, x' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_0 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:392
*/
struct TruncIShrSIToTrunciShrUI : public ::mlir::RewritePattern {
  TruncIShrSIToTrunciShrUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.trunci", 2, context, {"arith.constant", "arith.shrui", "arith.trunci"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::TruncIOp tr;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Attribute c0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::TruncIOp>(op0); (void)castedOp0;
    tr = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ShRSIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ShRSIOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
          ::mlir::Attribute arg2_0;
          if (!(!::mlir::failed(::mlir::success(::mlir::matchPattern(op2->getResult(0), ::mlir::m_Constant(&arg2_0)))))){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "::mlir::success(::mlir::matchPattern(op2->getResult(0), ::mlir::m_Constant(&arg2_0))) return ::mlir::failure";
            });
          }
          c0 = arg2_0;
          if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization5(rewriter, op2, arg2_0, "operand 0 of native code call '::mlir::success(::mlir::matchPattern($_self->getResult(0), ::mlir::m_Constant(&$0)))' failed to satisfy constraint: 'TypedAttr instance'"))) {
            return ::mlir::failure();
          }
        tblgen_ops.push_back(op2);
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((succeeded(getIntOrSplatIntValue(c0)))) && (((getScalarOrElementWidth((*x.begin())) - getScalarOrElementWidth((*tr.getODSResults(0).begin()))) == *getIntOrSplatIntValue(c0))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, tr, c0' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::cast<TypedAttr>(c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::ShRUIOp tblgen_ShRUIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_ShRUIOp_2 = rewriter.create<::mlir::arith::ShRUIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::TruncIOp tblgen_TruncIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ShRUIOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_TruncIOp_3 = rewriter.create<::mlir::arith::TruncIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_TruncIOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:399
*/
struct TruncIShrUIMulIToMulSIExtended : public ::mlir::RewritePattern {
  TruncIShrUIMulIToMulSIExtended(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.trunci", 5, context, {"arith.mulsi_extended"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::TruncIOp tr;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::MulIOp mul;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Attribute c0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::TruncIOp>(op0); (void)castedOp0;
    tr = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ShRUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ShRUIOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::arith::MulIOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::arith::MulIOp type";
          });
        }
        mul = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          if(::mlir::failed(static_dag_matcher_6(rewriter, op3, tblgen_ops, x))) {
            return ::mlir::failure();
          }
          tblgen_ops.push_back(op3);
        }
        {
          auto *op3 = (*castedOp2.getODSOperands(1).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 1 of castedOp2";
            });
          }
          if(::mlir::failed(static_dag_matcher_7(rewriter, op3, tblgen_ops, y))) {
            return ::mlir::failure();
          }
          tblgen_ops.push_back(op3);
        }
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
          if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
          ovf1 = tblgen_attr;
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_3(rewriter, op2, tblgen_ops, c0))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      tblgen_ops.push_back(op1);
    }
    if (!((llvm::all_equal({(*tr.getODSResults(0).begin()).getType(), (*x.begin()).getType(), (*y.begin()).getType()})))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'tr, x, y' failed to satisfy constraint: ''";
      });
    }
    if (!(((getScalarOrElementWidth((*mul.getODSResults(0).begin())) > getScalarOrElementWidth((*x.begin())))) && ((getScalarOrElementWidth((*x.begin())) > 0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'mul, x' failed to satisfy constraint: ''";
      });
    }
    if (!(((succeeded(getIntOrSplatIntValue(c0)))) && (((getScalarOrElementWidth((*mul.getODSResults(0).begin())) - getScalarOrElementWidth((*x.begin()))) == *getIntOrSplatIntValue(c0))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'mul, x, c0' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc(), tblgen_ops[4]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::MulSIExtendedOp res;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*y.begin());
      res = rewriter.create<::mlir::arith::MulSIExtendedOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ res.getODSResults(1) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:410
*/
struct TruncIShrUIMulIToMulUIExtended : public ::mlir::RewritePattern {
  TruncIShrUIMulIToMulUIExtended(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.trunci", 5, context, {"arith.mului_extended"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::TruncIOp tr;
    ::mlir::arith::IntegerOverflowFlagsAttr ovf1;
    ::mlir::arith::MulIOp mul;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Attribute c0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::TruncIOp>(op0); (void)castedOp0;
    tr = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ShRUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ShRUIOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::arith::MulIOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::arith::MulIOp type";
          });
        }
        mul = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          if(::mlir::failed(static_dag_matcher_8(rewriter, op3, tblgen_ops, x))) {
            return ::mlir::failure();
          }
          tblgen_ops.push_back(op3);
        }
        {
          auto *op3 = (*castedOp2.getODSOperands(1).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 1 of castedOp2";
            });
          }
          if(::mlir::failed(static_dag_matcher_9(rewriter, op3, tblgen_ops, y))) {
            return ::mlir::failure();
          }
          tblgen_ops.push_back(op3);
        }
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::arith::IntegerOverflowFlagsAttr>("overflowFlags");(void)tblgen_attr;
          if (!tblgen_attr) tblgen_attr = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none);
          ovf1 = tblgen_attr;
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_3(rewriter, op2, tblgen_ops, c0))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      tblgen_ops.push_back(op1);
    }
    if (!((llvm::all_equal({(*tr.getODSResults(0).begin()).getType(), (*x.begin()).getType(), (*y.begin()).getType()})))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'tr, x, y' failed to satisfy constraint: ''";
      });
    }
    if (!(((getScalarOrElementWidth((*mul.getODSResults(0).begin())) > getScalarOrElementWidth((*x.begin())))) && ((getScalarOrElementWidth((*x.begin())) > 0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'mul, x' failed to satisfy constraint: ''";
      });
    }
    if (!(((succeeded(getIntOrSplatIntValue(c0)))) && (((getScalarOrElementWidth((*mul.getODSResults(0).begin())) - getScalarOrElementWidth((*x.begin()))) == *getIntOrSplatIntValue(c0))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'mul, x, c0' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc(), tblgen_ops[4]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::MulUIExtendedOp res;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*y.begin());
      res = rewriter.create<::mlir::arith::MulUIExtendedOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ res.getODSResults(1) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:212
*/
struct XOrINotCmpI : public ::mlir::RewritePattern {
  XOrINotCmpI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.xori", 2, context, {"arith.cmpi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::CmpIPredicateAttr pred;
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::XOrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::CmpIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::CmpIOp type";
        });
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::CmpIPredicateAttr>("predicate");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'arith.cmpi' to have attribute 'predicate' of type '::mlir::arith::CmpIPredicateAttr'";
          });
        }
        pred = tblgen_attr;
      }
      a = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_12(rewriter, op1, tblgen_ops))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = invertPredicate(pred); (void)nativeVar_0;
    ::mlir::arith::CmpIOp tblgen_CmpIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpIOp_1 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:224
*/
struct XOrIOfExtSI : public ::mlir::RewritePattern {
  XOrIOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.xori", 3, context, {"arith.extsi", "arith.xori"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::XOrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_6(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::XOrIOp tblgen_XOrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_XOrIOp_0 = rewriter.create<::mlir::arith::XOrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_XOrIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_1 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:219
*/
struct XOrIOfExtUI : public ::mlir::RewritePattern {
  XOrIOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.xori", 3, context, {"arith.extui", "arith.xori"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::XOrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_9(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::XOrIOp tblgen_XOrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_XOrIOp_0 = rewriter.create<::mlir::arith::XOrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_XOrIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_1 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<AddIAddConstant>(patterns.getContext());
  patterns.add<AddIMulNegativeOneLhs>(patterns.getContext());
  patterns.add<AddIMulNegativeOneRhs>(patterns.getContext());
  patterns.add<AddISubConstantLHS>(patterns.getContext());
  patterns.add<AddISubConstantRHS>(patterns.getContext());
  patterns.add<AddUIExtendedToAddI>(patterns.getContext());
  patterns.add<AndOfExtSI>(patterns.getContext());
  patterns.add<AndOfExtUI>(patterns.getContext());
  patterns.add<BitcastOfBitcast>(patterns.getContext());
  patterns.add<CmpIExtSI>(patterns.getContext());
  patterns.add<CmpIExtUI>(patterns.getContext());
  patterns.add<DivFOfNegF>(patterns.getContext());
  patterns.add<ExtSIOfExtUI>(patterns.getContext());
  patterns.add<IndexCastOfExtSI>(patterns.getContext());
  patterns.add<IndexCastOfIndexCast>(patterns.getContext());
  patterns.add<IndexCastUIOfExtUI>(patterns.getContext());
  patterns.add<IndexCastUIOfIndexCastUI>(patterns.getContext());
  patterns.add<MulFOfNegF>(patterns.getContext());
  patterns.add<MulIMulIConstant>(patterns.getContext());
  patterns.add<MulSIExtendedRHSOne>(patterns.getContext());
  patterns.add<MulSIExtendedToMulI>(patterns.getContext());
  patterns.add<MulUIExtendedToMulI>(patterns.getContext());
  patterns.add<OrOfExtSI>(patterns.getContext());
  patterns.add<OrOfExtUI>(patterns.getContext());
  patterns.add<RedundantSelectFalse>(patterns.getContext());
  patterns.add<RedundantSelectTrue>(patterns.getContext());
  patterns.add<SelectI1ToNot>(patterns.getContext());
  patterns.add<SelectNotCond>(patterns.getContext());
  patterns.add<SubILHSAddConstant>(patterns.getContext());
  patterns.add<SubILHSSubConstantLHS>(patterns.getContext());
  patterns.add<SubILHSSubConstantRHS>(patterns.getContext());
  patterns.add<SubIRHSAddConstant>(patterns.getContext());
  patterns.add<SubIRHSSubConstantLHS>(patterns.getContext());
  patterns.add<SubIRHSSubConstantRHS>(patterns.getContext());
  patterns.add<SubISubILHSRHSLHS>(patterns.getContext());
  patterns.add<TruncIExtSIToExtSI>(patterns.getContext());
  patterns.add<TruncIExtUIToExtUI>(patterns.getContext());
  patterns.add<TruncIShrSIToTrunciShrUI>(patterns.getContext());
  patterns.add<TruncIShrUIMulIToMulSIExtended>(patterns.getContext());
  patterns.add<TruncIShrUIMulIToMulUIExtended>(patterns.getContext());
  patterns.add<XOrINotCmpI>(patterns.getContext());
  patterns.add<XOrIOfExtSI>(patterns.getContext());
  patterns.add<XOrIOfExtUI>(patterns.getContext());
}
