/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: optimize.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_optimize1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getBoolAttr(false)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute false";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_optimize2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr.cast<ElementsAttr>().getShapedType().getElementType().isF32()))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": float constant tensor";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_optimize3(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getStringAttr("NHWC")))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute \"NHWC\"";
    });
  }
  return ::mlir::success();
}
static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &bias) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
      });
    }
    if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize2(rewriter, op0, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'float constant tensor'"))) {
      return ::mlir::failure();
    }
    bias = tblgen_attr;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_1(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &value) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
      });
    }
    if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize2(rewriter, op0, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'float constant tensor'"))) {
      return ::mlir::failure();
    }
    value = tblgen_attr;
  }
  return ::mlir::success();
}

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:109
*/
struct ConvertCastComplexFFTToRFFT : public ::mlir::RewritePattern {
  ConvertCastComplexFFTToRFFT(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.FFT", 2, context, {"arith.constant", "tf.RFFT"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::FFTOp>(op0); (void)castedOp0;
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
      input = castedOp1.getODSOperands(0);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("Truncate");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
        if (!tblgen_attr) return ::mlir::failure();
        if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize1(rewriter, op1, tblgen_attr, "op 'tf.Cast' attribute 'Truncate' failed to satisfy constraint: 'constant attribute false'"))) {
          return ::mlir::failure();
        }
      }
      tblgen_ops.push_back(op1);
    }
    if (!((((*input.begin()).getType().isa<RankedTensorType>())) && ((!(*input.begin()).getType().cast<ShapedType>().isDynamicDim(   (*input.begin()).getType().cast<RankedTensorType>().getRank() - 1))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }
    if (!((((*input.begin()).getType().isa<RankedTensorType>())) && ((!(*input.begin()).getType().cast<ShapedType>().getElementType().isa<ComplexType>())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = DenseElementsAttr::get(RankedTensorType::get({1}, rewriter.getIntegerType(32)), static_cast<int32_t>((*input.begin()).getType().cast<RankedTensorType>().getDimSize(    (*input.begin()).getType().cast<RankedTensorType>().getRank() - 1))); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::RFFTOp tblgen_RFFTOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_RFFTOp_2 = rewriter.create<::mlir::TF::RFFTOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_RFFTOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:53
*/
struct FuseMulAndConv2D : public ::mlir::RewritePattern {
  FuseMulAndConv2D(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Mul", 4, context, {"arith.constant", "tf.Conv2D", "tf.Mul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::ArrayAttr strides;
    ::mlir::TypedAttr filter_value;
    ::mlir::TF::MulOp mul;
    ::mlir::BoolAttr use_cudnn;
    ::mlir::arith::ConstantOp filter;
    ::mlir::TF::Conv2DOp conv;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::StringAttr padding;
    ::mlir::ArrayAttr explicit_padding;
    ::mlir::StringAttr data_format;
    ::mlir::ArrayAttr dilations;
    ::mlir::arith::ConstantOp multiplier;
    ::mlir::TypedAttr mul_value;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MulOp>(op0); (void)castedOp0;
    mul = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::Conv2DOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::Conv2DOp type";
        });
      }
      conv = castedOp1;
      input = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::arith::ConstantOp type";
          });
        }
        filter = castedOp2;
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
          if (!(tblgen_attr)){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
            });
          }
          if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize2(rewriter, op2, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'float constant tensor'"))) {
            return ::mlir::failure();
          }
          filter_value = tblgen_attr;
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("strides");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.Conv2D' to have attribute 'strides' of type '::mlir::ArrayAttr'";
          });
        }
        strides = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("use_cudnn_on_gpu");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
        use_cudnn = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::StringAttr>("padding");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.Conv2D' to have attribute 'padding' of type '::mlir::StringAttr'";
          });
        }
        padding = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("explicit_paddings");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getI64ArrayAttr({});
        explicit_padding = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::StringAttr>("data_format");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("NHWC");
        if (!tblgen_attr) return ::mlir::failure();
        if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize3(rewriter, op1, tblgen_attr, "op 'tf.Conv2D' attribute 'data_format' failed to satisfy constraint: 'constant attribute \"NHWC\"'"))) {
          return ::mlir::failure();
        }
        data_format = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("dilations");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getI64ArrayAttr({1, 1, 1, 1});
        dilations = tblgen_attr;
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
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
        });
      }
      multiplier = castedOp1;
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
          });
        }
        if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize2(rewriter, op1, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'float constant tensor'"))) {
          return ::mlir::failure();
        }
        mul_value = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    if (!((TFL::IsBroadcastableElementsAttrs(filter_value, mul_value) && TFL::IsDimensionsDegenerateExceptLastOne(mul_value)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'filter_value, mul_value' failed to satisfy constraint: ''";
      });
    }
    if (!(((*conv.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'conv' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>((*filter.getODSResults(0).begin()).getLoc(),
        /*value=*/filter_value
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>((*multiplier.getODSResults(0).begin()).getLoc(),
        /*value=*/mul_value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstantOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_1.getODSResults(0).begin());
      tblgen_MulOp_2 = rewriter.create<::mlir::TF::MulOp>((*mul.getODSResults(0).begin()).getLoc(),
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::Conv2DOp tblgen_Conv2DOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_MulOp_2.getODSResults(0).begin()));
      if (auto tmpAttr = strides) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("strides"), tmpAttr);
      }
      if (auto tmpAttr = use_cudnn) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("use_cudnn_on_gpu"), tmpAttr);
      }
      if (auto tmpAttr = padding) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("padding"), tmpAttr);
      }
      if (auto tmpAttr = explicit_padding) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("explicit_paddings"), tmpAttr);
      }
      if (auto tmpAttr = data_format) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("data_format"), tmpAttr);
      }
      if (auto tmpAttr = dilations) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("dilations"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2DOp_3 = rewriter.create<::mlir::TF::Conv2DOp>((*conv.getODSResults(0).begin()).getLoc(), tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_Conv2DOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:99
*/
struct PassthroughMulAndAddV2 : public ::mlir::RewritePattern {
  PassthroughMulAndAddV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Mul", 4, context, {"arith.constant", "tf.AddV2", "tf.Mul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypedAttr bias;
    ::mlir::TF::AddV2Op output;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::TypedAttr value;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MulOp>(op0); (void)castedOp0;
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
      output = castedOp1;
      input = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, bias))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
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
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, value))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!((llvm::isa_and_nonnull<mlir::TF::Conv2DOp>((*input.begin()).getDefiningOp())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }
    if (!(((*output.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'output' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_0.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_2;
    {
      tblgen_ConstantOp_2 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/bias
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_3;
    {
      tblgen_ConstantOp_3 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstantOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_3.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_4.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddV2Op_5 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddV2Op_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:80
*/
struct PassthroughMulAndBiasAdd : public ::mlir::RewritePattern {
  PassthroughMulAndBiasAdd(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Mul", 4, context, {"arith.constant", "tf.AddV2", "tf.Mul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypedAttr bias;
    ::mlir::TF::BiasAddOp output;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::TypedAttr value;
    ::mlir::StringAttr format;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MulOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::BiasAddOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::BiasAddOp type";
        });
      }
      output = castedOp1;
      input = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, bias))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::StringAttr>("data_format");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("NHWC");
        if (!tblgen_attr) return ::mlir::failure();
        if(::mlir::failed(__mlir_ods_local_attr_constraint_optimize3(rewriter, op1, tblgen_attr, "op 'tf.BiasAdd' attribute 'data_format' failed to satisfy constraint: 'constant attribute \"NHWC\"'"))) {
          return ::mlir::failure();
        }
        format = tblgen_attr;
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
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, value))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!((llvm::isa_and_nonnull<mlir::TF::Conv2DOp>((*input.begin()).getDefiningOp())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: ''";
      });
    }
    if (!(((*output.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'output' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_0.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_2;
    {
      tblgen_ConstantOp_2 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/bias
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_3;
    {
      tblgen_ConstantOp_3 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstantOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_3.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_4.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddV2Op_5 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddV2Op_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertCastComplexFFTToRFFT>(patterns.getContext());
  patterns.add<FuseMulAndConv2D>(patterns.getContext());
  patterns.add<PassthroughMulAndAddV2>(patterns.getContext());
  patterns.add<PassthroughMulAndBiasAdd>(patterns.getContext());
}
