/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: VhloToVersionPatterns.td                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_VhloToVersionPatterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((isNoneType(attr)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": None type";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_VhloToVersionPatterns2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((isEmptyTensor(attr)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": Empty dims";
    });
  }
  return ::mlir::success();
}
/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:63
*/
struct AllGatherOpDowngradeV2ToV1 : public ::mlir::RewritePattern {
  AllGatherOpDowngradeV2ToV1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.all_gather_v2", 1, context, {"vhlo.all_gather_v1"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute all_gather_dim;
    ::mlir::Attribute use_global_device_ids;
    ::mlir::Attribute replica_groups;
    ::mlir::Attribute channel_id;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::AllGatherOpV2>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("all_gather_dim");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v2' to have attribute 'all_gather_dim' of type '::mlir::Attribute'";
        });
      }
      all_gather_dim = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("replica_groups");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v2' to have attribute 'replica_groups' of type '::mlir::Attribute'";
        });
      }
      replica_groups = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("channel_id");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v2' to have attribute 'channel_id' of type '::mlir::Attribute'";
        });
      }
      channel_id = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("use_global_device_ids");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v2' to have attribute 'use_global_device_ids' of type '::mlir::Attribute'";
        });
      }
      use_global_device_ids = tblgen_attr;
    }
    if (!((operand.size() == 1))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'operand' failed to satisfy constraint: 'Single operand'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = operand.front(); (void)nativeVar_0;
    ::mlir::vhlo::AllGatherOpV1 tblgen_AllGatherOpV1_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back(nativeVar_0);
      if (auto tmpAttr = all_gather_dim) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("all_gather_dim"), tmpAttr);
      }
      if (auto tmpAttr = replica_groups) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("replica_groups"), tmpAttr);
      }
      if (auto tmpAttr = channel_id) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("channel_id"), tmpAttr);
      }
      if (auto tmpAttr = use_global_device_ids) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("use_global_device_ids"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AllGatherOpV1_1 = rewriter.create<::mlir::vhlo::AllGatherOpV1>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AllGatherOpV1_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:59
*/
struct AllGatherOpUpgradeV1ToV2 : public ::mlir::RewritePattern {
  AllGatherOpUpgradeV1ToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.all_gather_v1", 1, context, {"vhlo.all_gather_v2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute all_gather_dim;
    ::mlir::Attribute use_global_device_ids;
    ::mlir::Attribute replica_groups;
    ::mlir::Attribute channel_id;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::AllGatherOpV1>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("all_gather_dim");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v1' to have attribute 'all_gather_dim' of type '::mlir::Attribute'";
        });
      }
      all_gather_dim = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("replica_groups");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v1' to have attribute 'replica_groups' of type '::mlir::Attribute'";
        });
      }
      replica_groups = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("channel_id");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v1' to have attribute 'channel_id' of type '::mlir::Attribute'";
        });
      }
      channel_id = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("use_global_device_ids");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_gather_v1' to have attribute 'use_global_device_ids' of type '::mlir::Attribute'";
        });
      }
      use_global_device_ids = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = {(*operand.begin())}; (void)nativeVar_0;
    ::mlir::vhlo::AllGatherOpV2 tblgen_AllGatherOpV2_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      for (auto v: nativeVar_0) {
        tblgen_values.push_back(v);
      }
      if (auto tmpAttr = all_gather_dim) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("all_gather_dim"), tmpAttr);
      }
      if (auto tmpAttr = replica_groups) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("replica_groups"), tmpAttr);
      }
      if (auto tmpAttr = channel_id) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("channel_id"), tmpAttr);
      }
      if (auto tmpAttr = use_global_device_ids) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("use_global_device_ids"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AllGatherOpV2_1 = rewriter.create<::mlir::vhlo::AllGatherOpV2>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AllGatherOpV2_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:72
*/
struct AllToAllOpDowngradeV2ToV1 : public ::mlir::RewritePattern {
  AllToAllOpDowngradeV2ToV1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.all_to_all_v2", 1, context, {"vhlo.all_to_all_v1"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute concat_dimension;
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute replica_groups;
    ::mlir::Attribute split_dimension;
    ::mlir::Attribute split_count;
    ::mlir::Attribute channel_id;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::AllToAllOpV2>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("split_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v2' to have attribute 'split_dimension' of type '::mlir::Attribute'";
        });
      }
      split_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("concat_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v2' to have attribute 'concat_dimension' of type '::mlir::Attribute'";
        });
      }
      concat_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("split_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v2' to have attribute 'split_count' of type '::mlir::Attribute'";
        });
      }
      split_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("replica_groups");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v2' to have attribute 'replica_groups' of type '::mlir::Attribute'";
        });
      }
      replica_groups = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("channel_id");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v2' to have attribute 'channel_id' of type '::mlir::Attribute'";
        });
      }
      channel_id = tblgen_attr;
    }
    if (!((operand.size() == 1))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'operand' failed to satisfy constraint: 'Single operand'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = operand.front(); (void)nativeVar_0;
    ::mlir::vhlo::AllToAllOpV1 tblgen_AllToAllOpV1_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back(nativeVar_0);
      if (auto tmpAttr = split_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("split_dimension"), tmpAttr);
      }
      if (auto tmpAttr = concat_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("concat_dimension"), tmpAttr);
      }
      if (auto tmpAttr = split_count) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("split_count"), tmpAttr);
      }
      if (auto tmpAttr = replica_groups) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("replica_groups"), tmpAttr);
      }
      if (auto tmpAttr = channel_id) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("channel_id"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AllToAllOpV1_1 = rewriter.create<::mlir::vhlo::AllToAllOpV1>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AllToAllOpV1_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:68
*/
struct AllToAllOpUpgradeV1ToV2 : public ::mlir::RewritePattern {
  AllToAllOpUpgradeV1ToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.all_to_all_v1", 1, context, {"vhlo.all_to_all_v2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute concat_dimension;
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute replica_groups;
    ::mlir::Attribute split_dimension;
    ::mlir::Attribute split_count;
    ::mlir::Attribute channel_id;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::AllToAllOpV1>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("split_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v1' to have attribute 'split_dimension' of type '::mlir::Attribute'";
        });
      }
      split_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("concat_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v1' to have attribute 'concat_dimension' of type '::mlir::Attribute'";
        });
      }
      concat_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("split_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v1' to have attribute 'split_count' of type '::mlir::Attribute'";
        });
      }
      split_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("replica_groups");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v1' to have attribute 'replica_groups' of type '::mlir::Attribute'";
        });
      }
      replica_groups = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("channel_id");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.all_to_all_v1' to have attribute 'channel_id' of type '::mlir::Attribute'";
        });
      }
      channel_id = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = {(*operand.begin())}; (void)nativeVar_0;
    ::mlir::vhlo::AllToAllOpV2 tblgen_AllToAllOpV2_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      for (auto v: nativeVar_0) {
        tblgen_values.push_back(v);
      }
      if (auto tmpAttr = split_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("split_dimension"), tmpAttr);
      }
      if (auto tmpAttr = concat_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("concat_dimension"), tmpAttr);
      }
      if (auto tmpAttr = split_count) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("split_count"), tmpAttr);
      }
      if (auto tmpAttr = replica_groups) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("replica_groups"), tmpAttr);
      }
      if (auto tmpAttr = channel_id) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("channel_id"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AllToAllOpV2_1 = rewriter.create<::mlir::vhlo::AllToAllOpV2>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AllToAllOpV2_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:77
*/
struct DotGeneralOpDowngradeV2ToV1 : public ::mlir::RewritePattern {
  DotGeneralOpDowngradeV2ToV1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.dot_general_v2", 1, context, {"vhlo.dot_general_v1"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute lhs_precision_type;
    ::mlir::Attribute precision_config;
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Attribute lhs_contracting_dimensions;
    ::mlir::Attribute lhs_batching_dimensions;
    ::mlir::Attribute rhs_contracting_dimensions;
    ::mlir::Attribute rhs_batching_dimensions;
    ::mlir::Attribute allow_imprecise_accumulation;
    ::mlir::Attribute rhs_precision_type;
    ::mlir::Attribute accumulation_type;
    ::mlir::Attribute lhs_component_count;
    ::mlir::Attribute rhs_component_count;
    ::mlir::Attribute num_primitive_operations;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::DotGeneralOpV2>(op0); (void)castedOp0;
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_batching_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'lhs_batching_dimensions' of type '::mlir::Attribute'";
        });
      }
      lhs_batching_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_batching_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'rhs_batching_dimensions' of type '::mlir::Attribute'";
        });
      }
      rhs_batching_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_contracting_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'lhs_contracting_dimensions' of type '::mlir::Attribute'";
        });
      }
      lhs_contracting_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_contracting_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'rhs_contracting_dimensions' of type '::mlir::Attribute'";
        });
      }
      rhs_contracting_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("precision_config");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'precision_config' of type '::mlir::Attribute'";
        });
      }
      precision_config = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_precision_type");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'lhs_precision_type' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'lhs_precision_type' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      lhs_precision_type = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_precision_type");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'rhs_precision_type' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'rhs_precision_type' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      rhs_precision_type = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("accumulation_type");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'accumulation_type' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'accumulation_type' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      accumulation_type = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_component_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'lhs_component_count' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'lhs_component_count' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      lhs_component_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_component_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'rhs_component_count' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'rhs_component_count' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      rhs_component_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("num_primitive_operations");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'num_primitive_operations' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'num_primitive_operations' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      num_primitive_operations = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("allow_imprecise_accumulation");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v2' to have attribute 'allow_imprecise_accumulation' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns1(rewriter, op0, tblgen_attr, "op 'vhlo.dot_general_v2' attribute 'allow_imprecise_accumulation' failed to satisfy constraint: 'None type'"))) {
        return ::mlir::failure();
      }
      allow_imprecise_accumulation = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::vhlo::DotGeneralOpV1 tblgen_DotGeneralOpV1_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      if (auto tmpAttr = lhs_batching_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_batching_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = rhs_batching_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_batching_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = lhs_contracting_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_contracting_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = rhs_contracting_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_contracting_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = precision_config) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("precision_config"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DotGeneralOpV1_0 = rewriter.create<::mlir::vhlo::DotGeneralOpV1>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DotGeneralOpV1_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:82
*/
struct DotGeneralOpUpradeV1ToV2 : public ::mlir::RewritePattern {
  DotGeneralOpUpradeV1ToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.dot_general_v1", 1, context, {"vhlo.dot_general_v2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute precision_config;
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Attribute lhs_contracting_dimensions;
    ::mlir::Attribute lhs_batching_dimensions;
    ::mlir::Attribute rhs_contracting_dimensions;
    ::mlir::Attribute rhs_batching_dimensions;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::DotGeneralOpV1>(op0); (void)castedOp0;
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_batching_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v1' to have attribute 'lhs_batching_dimensions' of type '::mlir::Attribute'";
        });
      }
      lhs_batching_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_batching_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v1' to have attribute 'rhs_batching_dimensions' of type '::mlir::Attribute'";
        });
      }
      rhs_batching_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_contracting_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v1' to have attribute 'lhs_contracting_dimensions' of type '::mlir::Attribute'";
        });
      }
      lhs_contracting_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_contracting_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v1' to have attribute 'rhs_contracting_dimensions' of type '::mlir::Attribute'";
        });
      }
      rhs_contracting_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("precision_config");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dot_general_v1' to have attribute 'precision_config' of type '::mlir::Attribute'";
        });
      }
      precision_config = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = getNoneType(rewriter); (void)nativeVar_0;
    auto nativeVar_1 = getNoneType(rewriter); (void)nativeVar_1;
    auto nativeVar_2 = getNoneType(rewriter); (void)nativeVar_2;
    auto nativeVar_3 = getNoneType(rewriter); (void)nativeVar_3;
    auto nativeVar_4 = getNoneType(rewriter); (void)nativeVar_4;
    auto nativeVar_5 = getNoneType(rewriter); (void)nativeVar_5;
    auto nativeVar_6 = getNoneType(rewriter); (void)nativeVar_6;
    ::mlir::vhlo::DotGeneralOpV2 tblgen_DotGeneralOpV2_7;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      if (auto tmpAttr = lhs_batching_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_batching_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = rhs_batching_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_batching_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = lhs_contracting_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_contracting_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = rhs_contracting_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_contracting_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = precision_config) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("precision_config"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_precision_type"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_precision_type"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("accumulation_type"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_3) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_component_count"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_4) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_component_count"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_5) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("num_primitive_operations"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_6) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("allow_imprecise_accumulation"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DotGeneralOpV2_7 = rewriter.create<::mlir::vhlo::DotGeneralOpV2>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DotGeneralOpV2_7.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:39
*/
struct DynamicConvDowngradeV2ToV1 : public ::mlir::RewritePattern {
  DynamicConvDowngradeV2ToV1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.dynamic_conv_v2", 1, context, {"vhlo.dynamic_conv_v1"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Attribute input_spatial_dimensions;
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range d_padding(op0->getOperands());
    ::mlir::Attribute window_reversal;
    ::mlir::Attribute window_strides;
    ::mlir::Attribute lhs_dilation;
    ::mlir::Attribute rhs_dilation;
    ::mlir::Attribute input_batch_dimension;
    ::mlir::Attribute feature_group_count;
    ::mlir::Attribute kernel_input_feature_dimension;
    ::mlir::Attribute input_feature_dimension;
    ::mlir::Attribute output_feature_dimension;
    ::mlir::Attribute kernel_output_feature_dimension;
    ::mlir::Attribute kernel_spatial_dimensions;
    ::mlir::Attribute output_batch_dimension;
    ::mlir::Attribute output_spatial_dimensions;
    ::mlir::Attribute batch_group_count;
    ::mlir::Attribute precision_config;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::DynamicConvOpV2>(op0); (void)castedOp0;
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);
    d_padding = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("window_strides");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'window_strides' of type '::mlir::Attribute'";
        });
      }
      window_strides = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_dilation");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'lhs_dilation' of type '::mlir::Attribute'";
        });
      }
      lhs_dilation = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_dilation");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'rhs_dilation' of type '::mlir::Attribute'";
        });
      }
      rhs_dilation = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("window_reversal");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'window_reversal' of type '::mlir::Attribute'";
        });
      }
      window_reversal = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("input_batch_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'input_batch_dimension' of type '::mlir::Attribute'";
        });
      }
      input_batch_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("input_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'input_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      input_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("input_spatial_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'input_spatial_dimensions' of type '::mlir::Attribute'";
        });
      }
      input_spatial_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("kernel_input_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'kernel_input_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      kernel_input_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("kernel_output_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'kernel_output_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      kernel_output_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("kernel_spatial_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'kernel_spatial_dimensions' of type '::mlir::Attribute'";
        });
      }
      kernel_spatial_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("output_batch_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'output_batch_dimension' of type '::mlir::Attribute'";
        });
      }
      output_batch_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("output_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'output_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      output_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("output_spatial_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'output_spatial_dimensions' of type '::mlir::Attribute'";
        });
      }
      output_spatial_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("feature_group_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'feature_group_count' of type '::mlir::Attribute'";
        });
      }
      feature_group_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("batch_group_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'batch_group_count' of type '::mlir::Attribute'";
        });
      }
      batch_group_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("precision_config");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v2' to have attribute 'precision_config' of type '::mlir::Attribute'";
        });
      }
      precision_config = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = getDefaultConvPadding(rewriter, (*lhs.begin())); (void)nativeVar_0;
    ::mlir::vhlo::DynamicConvOpV1 tblgen_DynamicConvOpV1_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      tblgen_values.push_back((*d_padding.begin()));
      if (auto tmpAttr = window_strides) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("window_strides"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("padding"), tmpAttr);
      }
      if (auto tmpAttr = lhs_dilation) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_dilation"), tmpAttr);
      }
      if (auto tmpAttr = rhs_dilation) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_dilation"), tmpAttr);
      }
      if (auto tmpAttr = window_reversal) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("window_reversal"), tmpAttr);
      }
      if (auto tmpAttr = input_batch_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("input_batch_dimension"), tmpAttr);
      }
      if (auto tmpAttr = input_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("input_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = input_spatial_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("input_spatial_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = kernel_input_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("kernel_input_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = kernel_output_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("kernel_output_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = kernel_spatial_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("kernel_spatial_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = output_batch_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("output_batch_dimension"), tmpAttr);
      }
      if (auto tmpAttr = output_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("output_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = output_spatial_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("output_spatial_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = feature_group_count) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("feature_group_count"), tmpAttr);
      }
      if (auto tmpAttr = batch_group_count) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("batch_group_count"), tmpAttr);
      }
      if (auto tmpAttr = precision_config) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("precision_config"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DynamicConvOpV1_1 = rewriter.create<::mlir::vhlo::DynamicConvOpV1>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DynamicConvOpV1_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:35
*/
struct DynamicConvUpgradeV1ToV2 : public ::mlir::RewritePattern {
  DynamicConvUpgradeV1ToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.dynamic_conv_v1", 1, context, {"vhlo.dynamic_conv_v2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute precision_config;
    ::mlir::Attribute padding;
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Attribute input_spatial_dimensions;
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range d_padding(op0->getOperands());
    ::mlir::Attribute window_reversal;
    ::mlir::Attribute window_strides;
    ::mlir::Attribute lhs_dilation;
    ::mlir::Attribute rhs_dilation;
    ::mlir::Attribute input_batch_dimension;
    ::mlir::Attribute feature_group_count;
    ::mlir::Attribute kernel_input_feature_dimension;
    ::mlir::Attribute input_feature_dimension;
    ::mlir::Attribute output_feature_dimension;
    ::mlir::Attribute kernel_output_feature_dimension;
    ::mlir::Attribute kernel_spatial_dimensions;
    ::mlir::Attribute output_batch_dimension;
    ::mlir::Attribute output_spatial_dimensions;
    ::mlir::Attribute batch_group_count;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::DynamicConvOpV1>(op0); (void)castedOp0;
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);
    d_padding = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("window_strides");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'window_strides' of type '::mlir::Attribute'";
        });
      }
      window_strides = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("padding");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'padding' of type '::mlir::Attribute'";
        });
      }
      padding = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("lhs_dilation");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'lhs_dilation' of type '::mlir::Attribute'";
        });
      }
      lhs_dilation = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("rhs_dilation");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'rhs_dilation' of type '::mlir::Attribute'";
        });
      }
      rhs_dilation = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("window_reversal");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'window_reversal' of type '::mlir::Attribute'";
        });
      }
      window_reversal = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("input_batch_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'input_batch_dimension' of type '::mlir::Attribute'";
        });
      }
      input_batch_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("input_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'input_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      input_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("input_spatial_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'input_spatial_dimensions' of type '::mlir::Attribute'";
        });
      }
      input_spatial_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("kernel_input_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'kernel_input_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      kernel_input_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("kernel_output_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'kernel_output_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      kernel_output_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("kernel_spatial_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'kernel_spatial_dimensions' of type '::mlir::Attribute'";
        });
      }
      kernel_spatial_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("output_batch_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'output_batch_dimension' of type '::mlir::Attribute'";
        });
      }
      output_batch_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("output_feature_dimension");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'output_feature_dimension' of type '::mlir::Attribute'";
        });
      }
      output_feature_dimension = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("output_spatial_dimensions");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'output_spatial_dimensions' of type '::mlir::Attribute'";
        });
      }
      output_spatial_dimensions = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("feature_group_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'feature_group_count' of type '::mlir::Attribute'";
        });
      }
      feature_group_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("batch_group_count");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'batch_group_count' of type '::mlir::Attribute'";
        });
      }
      batch_group_count = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("precision_config");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_conv_v1' to have attribute 'precision_config' of type '::mlir::Attribute'";
        });
      }
      precision_config = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::vhlo::DynamicConvOpV2 tblgen_DynamicConvOpV2_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      tblgen_values.push_back((*d_padding.begin()));
      if (auto tmpAttr = window_strides) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("window_strides"), tmpAttr);
      }
      if (auto tmpAttr = lhs_dilation) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("lhs_dilation"), tmpAttr);
      }
      if (auto tmpAttr = rhs_dilation) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("rhs_dilation"), tmpAttr);
      }
      if (auto tmpAttr = window_reversal) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("window_reversal"), tmpAttr);
      }
      if (auto tmpAttr = input_batch_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("input_batch_dimension"), tmpAttr);
      }
      if (auto tmpAttr = input_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("input_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = input_spatial_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("input_spatial_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = kernel_input_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("kernel_input_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = kernel_output_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("kernel_output_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = kernel_spatial_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("kernel_spatial_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = output_batch_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("output_batch_dimension"), tmpAttr);
      }
      if (auto tmpAttr = output_feature_dimension) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("output_feature_dimension"), tmpAttr);
      }
      if (auto tmpAttr = output_spatial_dimensions) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("output_spatial_dimensions"), tmpAttr);
      }
      if (auto tmpAttr = feature_group_count) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("feature_group_count"), tmpAttr);
      }
      if (auto tmpAttr = batch_group_count) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("batch_group_count"), tmpAttr);
      }
      if (auto tmpAttr = precision_config) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("precision_config"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DynamicConvOpV2_0 = rewriter.create<::mlir::vhlo::DynamicConvOpV2>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DynamicConvOpV2_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:55
*/
struct DynamicGatherOpDowngradeV2ToV1 : public ::mlir::RewritePattern {
  DynamicGatherOpDowngradeV2ToV1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.dynamic_gather_v2", 1, context, {"vhlo.dynamic_gather_v1"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute start_index_map;
    ::mlir::Operation::operand_range start_indices(op0->getOperands());
    ::mlir::Attribute start_indices_batching_dims;
    ::mlir::Operation::operand_range slice_sizes(op0->getOperands());
    ::mlir::Attribute offset_dims;
    ::mlir::Attribute operand_batching_dims;
    ::mlir::Attribute collapsed_slice_dims;
    ::mlir::Attribute index_vector_dim;
    ::mlir::Attribute indices_are_sorted;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::DynamicGatherOpV2>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    start_indices = castedOp0.getODSOperands(1);
    slice_sizes = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("offset_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'offset_dims' of type '::mlir::Attribute'";
        });
      }
      offset_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("collapsed_slice_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'collapsed_slice_dims' of type '::mlir::Attribute'";
        });
      }
      collapsed_slice_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("operand_batching_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'operand_batching_dims' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns2(rewriter, op0, tblgen_attr, "op 'vhlo.dynamic_gather_v2' attribute 'operand_batching_dims' failed to satisfy constraint: 'Empty dims'"))) {
        return ::mlir::failure();
      }
      operand_batching_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("start_indices_batching_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'start_indices_batching_dims' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns2(rewriter, op0, tblgen_attr, "op 'vhlo.dynamic_gather_v2' attribute 'start_indices_batching_dims' failed to satisfy constraint: 'Empty dims'"))) {
        return ::mlir::failure();
      }
      start_indices_batching_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("start_index_map");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'start_index_map' of type '::mlir::Attribute'";
        });
      }
      start_index_map = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("index_vector_dim");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'index_vector_dim' of type '::mlir::Attribute'";
        });
      }
      index_vector_dim = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("indices_are_sorted");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v2' to have attribute 'indices_are_sorted' of type '::mlir::Attribute'";
        });
      }
      indices_are_sorted = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::vhlo::DynamicGatherOpV1 tblgen_DynamicGatherOpV1_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*operand.begin()));
      tblgen_values.push_back((*start_indices.begin()));
      tblgen_values.push_back((*slice_sizes.begin()));
      if (auto tmpAttr = offset_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("offset_dims"), tmpAttr);
      }
      if (auto tmpAttr = collapsed_slice_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("collapsed_slice_dims"), tmpAttr);
      }
      if (auto tmpAttr = start_index_map) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("start_index_map"), tmpAttr);
      }
      if (auto tmpAttr = index_vector_dim) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("index_vector_dim"), tmpAttr);
      }
      if (auto tmpAttr = indices_are_sorted) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("indices_are_sorted"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DynamicGatherOpV1_0 = rewriter.create<::mlir::vhlo::DynamicGatherOpV1>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DynamicGatherOpV1_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:51
*/
struct DynamicGatherOpUpgradeV1ToV2 : public ::mlir::RewritePattern {
  DynamicGatherOpUpgradeV1ToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.dynamic_gather_v1", 1, context, {"vhlo.dynamic_gather_v2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute indices_are_sorted;
    ::mlir::Attribute start_index_map;
    ::mlir::Operation::operand_range start_indices(op0->getOperands());
    ::mlir::Operation::operand_range slice_sizes(op0->getOperands());
    ::mlir::Attribute offset_dims;
    ::mlir::Attribute collapsed_slice_dims;
    ::mlir::Attribute index_vector_dim;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::DynamicGatherOpV1>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    start_indices = castedOp0.getODSOperands(1);
    slice_sizes = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("offset_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v1' to have attribute 'offset_dims' of type '::mlir::Attribute'";
        });
      }
      offset_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("collapsed_slice_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v1' to have attribute 'collapsed_slice_dims' of type '::mlir::Attribute'";
        });
      }
      collapsed_slice_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("start_index_map");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v1' to have attribute 'start_index_map' of type '::mlir::Attribute'";
        });
      }
      start_index_map = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("index_vector_dim");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v1' to have attribute 'index_vector_dim' of type '::mlir::Attribute'";
        });
      }
      index_vector_dim = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("indices_are_sorted");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.dynamic_gather_v1' to have attribute 'indices_are_sorted' of type '::mlir::Attribute'";
        });
      }
      indices_are_sorted = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = getEmptyI64Tensor(rewriter); (void)nativeVar_0;
    auto nativeVar_1 = getEmptyI64Tensor(rewriter); (void)nativeVar_1;
    ::mlir::vhlo::DynamicGatherOpV2 tblgen_DynamicGatherOpV2_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*operand.begin()));
      tblgen_values.push_back((*start_indices.begin()));
      tblgen_values.push_back((*slice_sizes.begin()));
      if (auto tmpAttr = offset_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("offset_dims"), tmpAttr);
      }
      if (auto tmpAttr = collapsed_slice_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("collapsed_slice_dims"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("operand_batching_dims"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("start_indices_batching_dims"), tmpAttr);
      }
      if (auto tmpAttr = start_index_map) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("start_index_map"), tmpAttr);
      }
      if (auto tmpAttr = index_vector_dim) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("index_vector_dim"), tmpAttr);
      }
      if (auto tmpAttr = indices_are_sorted) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("indices_are_sorted"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DynamicGatherOpV2_2 = rewriter.create<::mlir::vhlo::DynamicGatherOpV2>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DynamicGatherOpV2_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:47
*/
struct GatherOpDowngradeV2ToV1 : public ::mlir::RewritePattern {
  GatherOpDowngradeV2ToV1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.gather_v2", 1, context, {"vhlo.gather_v1"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute start_index_map;
    ::mlir::Operation::operand_range start_indices(op0->getOperands());
    ::mlir::Attribute offset_dims;
    ::mlir::Attribute operand_batching_dims;
    ::mlir::Attribute collapsed_slice_dims;
    ::mlir::Attribute start_indices_batching_dims;
    ::mlir::Attribute index_vector_dim;
    ::mlir::Attribute slice_sizes;
    ::mlir::Attribute indices_are_sorted;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::GatherOpV2>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    start_indices = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("offset_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'offset_dims' of type '::mlir::Attribute'";
        });
      }
      offset_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("collapsed_slice_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'collapsed_slice_dims' of type '::mlir::Attribute'";
        });
      }
      collapsed_slice_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("operand_batching_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'operand_batching_dims' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns2(rewriter, op0, tblgen_attr, "op 'vhlo.gather_v2' attribute 'operand_batching_dims' failed to satisfy constraint: 'Empty dims'"))) {
        return ::mlir::failure();
      }
      operand_batching_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("start_indices_batching_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'start_indices_batching_dims' of type '::mlir::Attribute'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_VhloToVersionPatterns2(rewriter, op0, tblgen_attr, "op 'vhlo.gather_v2' attribute 'start_indices_batching_dims' failed to satisfy constraint: 'Empty dims'"))) {
        return ::mlir::failure();
      }
      start_indices_batching_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("start_index_map");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'start_index_map' of type '::mlir::Attribute'";
        });
      }
      start_index_map = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("index_vector_dim");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'index_vector_dim' of type '::mlir::Attribute'";
        });
      }
      index_vector_dim = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("slice_sizes");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'slice_sizes' of type '::mlir::Attribute'";
        });
      }
      slice_sizes = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("indices_are_sorted");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v2' to have attribute 'indices_are_sorted' of type '::mlir::Attribute'";
        });
      }
      indices_are_sorted = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::vhlo::GatherOpV1 tblgen_GatherOpV1_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*operand.begin()));
      tblgen_values.push_back((*start_indices.begin()));
      if (auto tmpAttr = offset_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("offset_dims"), tmpAttr);
      }
      if (auto tmpAttr = collapsed_slice_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("collapsed_slice_dims"), tmpAttr);
      }
      if (auto tmpAttr = start_index_map) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("start_index_map"), tmpAttr);
      }
      if (auto tmpAttr = index_vector_dim) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("index_vector_dim"), tmpAttr);
      }
      if (auto tmpAttr = slice_sizes) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("slice_sizes"), tmpAttr);
      }
      if (auto tmpAttr = indices_are_sorted) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("indices_are_sorted"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GatherOpV1_0 = rewriter.create<::mlir::vhlo::GatherOpV1>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GatherOpV1_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/stablehlo/stablehlo/transforms/VhloToVersionPatterns.td:43
*/
struct GatherOpUpgradeV1ToV2 : public ::mlir::RewritePattern {
  GatherOpUpgradeV1ToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("vhlo.gather_v1", 1, context, {"vhlo.gather_v2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::mlir::Attribute indices_are_sorted;
    ::mlir::Attribute start_index_map;
    ::mlir::Operation::operand_range start_indices(op0->getOperands());
    ::mlir::Attribute offset_dims;
    ::mlir::Attribute collapsed_slice_dims;
    ::mlir::Attribute index_vector_dim;
    ::mlir::Attribute slice_sizes;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::vhlo::GatherOpV1>(op0); (void)castedOp0;
    operand = castedOp0.getODSOperands(0);
    start_indices = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("offset_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v1' to have attribute 'offset_dims' of type '::mlir::Attribute'";
        });
      }
      offset_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("collapsed_slice_dims");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v1' to have attribute 'collapsed_slice_dims' of type '::mlir::Attribute'";
        });
      }
      collapsed_slice_dims = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("start_index_map");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v1' to have attribute 'start_index_map' of type '::mlir::Attribute'";
        });
      }
      start_index_map = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("index_vector_dim");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v1' to have attribute 'index_vector_dim' of type '::mlir::Attribute'";
        });
      }
      index_vector_dim = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("slice_sizes");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v1' to have attribute 'slice_sizes' of type '::mlir::Attribute'";
        });
      }
      slice_sizes = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("indices_are_sorted");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'vhlo.gather_v1' to have attribute 'indices_are_sorted' of type '::mlir::Attribute'";
        });
      }
      indices_are_sorted = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = getEmptyI64Tensor(rewriter); (void)nativeVar_0;
    auto nativeVar_1 = getEmptyI64Tensor(rewriter); (void)nativeVar_1;
    ::mlir::vhlo::GatherOpV2 tblgen_GatherOpV2_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*operand.begin()));
      tblgen_values.push_back((*start_indices.begin()));
      if (auto tmpAttr = offset_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("offset_dims"), tmpAttr);
      }
      if (auto tmpAttr = collapsed_slice_dims) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("collapsed_slice_dims"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("operand_batching_dims"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_1) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("start_indices_batching_dims"), tmpAttr);
      }
      if (auto tmpAttr = start_index_map) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("start_index_map"), tmpAttr);
      }
      if (auto tmpAttr = index_vector_dim) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("index_vector_dim"), tmpAttr);
      }
      if (auto tmpAttr = slice_sizes) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("slice_sizes"), tmpAttr);
      }
      if (auto tmpAttr = indices_are_sorted) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("indices_are_sorted"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GatherOpV2_2 = rewriter.create<::mlir::vhlo::GatherOpV2>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GatherOpV2_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<AllGatherOpDowngradeV2ToV1>(patterns.getContext());
  patterns.add<AllGatherOpUpgradeV1ToV2>(patterns.getContext());
  patterns.add<AllToAllOpDowngradeV2ToV1>(patterns.getContext());
  patterns.add<AllToAllOpUpgradeV1ToV2>(patterns.getContext());
  patterns.add<DotGeneralOpDowngradeV2ToV1>(patterns.getContext());
  patterns.add<DotGeneralOpUpradeV1ToV2>(patterns.getContext());
  patterns.add<DynamicConvDowngradeV2ToV1>(patterns.getContext());
  patterns.add<DynamicConvUpgradeV1ToV2>(patterns.getContext());
  patterns.add<DynamicGatherOpDowngradeV2ToV1>(patterns.getContext());
  patterns.add<DynamicGatherOpUpgradeV1ToV2>(patterns.getContext());
  patterns.add<GatherOpDowngradeV2ToV1>(patterns.getContext());
  patterns.add<GatherOpUpgradeV1ToV2>(patterns.getContext());
}
