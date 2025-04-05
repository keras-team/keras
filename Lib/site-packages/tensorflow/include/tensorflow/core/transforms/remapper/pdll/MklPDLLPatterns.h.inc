static ::llvm::LogicalResult AttrIsF32OrBF16PDLFn(::mlir::PatternRewriter &rewriter, ::mlir::Attribute attr) {
  TypeAttr type_attr = attr.dyn_cast<TypeAttr>();
  if (!type_attr) return failure();
  return success(type_attr.getValue().isa<Float32Type>() ||
                 type_attr.getValue().isa<BFloat16Type>());
}

static ::mlir::Operation * ReplaceMulWith_MklSwishPDLFn(::mlir::PatternRewriter &rewriter, ::mlir::Operation * op, ::mlir::Value arg, ::mlir::ValueRange controls) {
  SmallVector<Value> operands;
  operands.push_back(arg);
  llvm::append_range(operands, controls);
  Operation *new_op = rewriter.create(op->getLoc(),
                                      rewriter.getStringAttr("tfg._MklSwish"),
                                      operands,
                                      op->getResultTypes(),
                                      op->getAttrs());
  return new_op;
}

namespace {

struct SigmoidAndMul0 : ::mlir::PDLPatternModule {
  template <typename... ConfigsT>
  SigmoidAndMul0(::mlir::MLIRContext *context, ConfigsT &&...configs)
    : ::mlir::PDLPatternModule(::mlir::parseSourceString<::mlir::ModuleOp>(
R"mlir(pdl.pattern @SigmoidAndMul0 : benefit(0) {
  %0 = operand loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":40:37)
  %1 = types loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":40:21)
  %2 = operation "tfg.Sigmoid"(%0 : !pdl.value)  -> (%1 : !pdl.range<type>) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":40:21)
  %3 = result 0 of %2 loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":41:26)
  %4 = operands loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":41:46)
  %5 = attribute loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":41:76)
  apply_native_constraint "AttrIsF32OrBF16"(%5 : !pdl.attribute) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":41:79)
  %6 = types loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":41:14)
  %7 = operation "tfg.Mul"(%3, %0, %4 : !pdl.value, !pdl.value, !pdl.range<value>)  {"T" = %5} -> (%6 : !pdl.range<type>) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":41:14)
  apply_native_constraint "OpHasCpuDevice"(%7 : !pdl.operation) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":42:3)
  rewrite %7 {
    %8 = apply_native_rewrite "ReplaceMulWith_MklSwish"(%7, %0, %4 : !pdl.operation, !pdl.value, !pdl.range<value>) : !pdl.operation loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":43:21)
    replace %7 with %8 loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":43:3)
  } loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":43:3)
} loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":38:1)
    )mlir", context), std::forward<ConfigsT>(configs)...) {
    registerConstraintFunction("AttrIsF32OrBF16", AttrIsF32OrBF16PDLFn);
    registerRewriteFunction("ReplaceMulWith_MklSwish", ReplaceMulWith_MklSwishPDLFn);
  }
};


struct SigmoidAndMul1 : ::mlir::PDLPatternModule {
  template <typename... ConfigsT>
  SigmoidAndMul1(::mlir::MLIRContext *context, ConfigsT &&...configs)
    : ::mlir::PDLPatternModule(::mlir::parseSourceString<::mlir::ModuleOp>(
R"mlir(pdl.pattern @SigmoidAndMul1 : benefit(0) {
  %0 = operand loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":49:37)
  %1 = types loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":49:21)
  %2 = operation "tfg.Sigmoid"(%0 : !pdl.value)  -> (%1 : !pdl.range<type>) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":49:21)
  %3 = result 0 of %2 loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":50:31)
  %4 = operands loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":50:46)
  %5 = attribute loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":50:76)
  apply_native_constraint "AttrIsF32OrBF16"(%5 : !pdl.attribute) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":50:79)
  %6 = types loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":50:14)
  %7 = operation "tfg.Mul"(%0, %3, %4 : !pdl.value, !pdl.value, !pdl.range<value>)  {"T" = %5} -> (%6 : !pdl.range<type>) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":50:14)
  apply_native_constraint "OpHasCpuDevice"(%7 : !pdl.operation) loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":51:3)
  rewrite %7 {
    %8 = apply_native_rewrite "ReplaceMulWith_MklSwish"(%7, %0, %4 : !pdl.operation, !pdl.value, !pdl.range<value>) : !pdl.operation loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":52:21)
    replace %7 with %8 loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":52:3)
  } loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":52:3)
} loc("tensorflow/core/transforms/remapper/pdll/mkl_patterns.pdll":47:1)
    )mlir", context), std::forward<ConfigsT>(configs)...) {
    registerConstraintFunction("AttrIsF32OrBF16", AttrIsF32OrBF16PDLFn);
    registerRewriteFunction("ReplaceMulWith_MklSwish", ReplaceMulWith_MklSwishPDLFn);
  }
};

} // end namespace

template <typename... ConfigsT>
static void LLVM_ATTRIBUTE_UNUSED populateGeneratedPDLLPatterns(::mlir::RewritePatternSet &patterns, ConfigsT &&...configs) {
  patterns.add<SigmoidAndMul0>(patterns.getContext(), configs...);
  patterns.add<SigmoidAndMul1>(patterns.getContext(), configs...);
}
