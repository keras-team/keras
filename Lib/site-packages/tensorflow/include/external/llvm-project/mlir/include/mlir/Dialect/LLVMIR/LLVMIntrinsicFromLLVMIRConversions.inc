if (intrinsicID == llvm::Intrinsic::abs) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1},
      {StringLiteral("is_int_min_poison")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::AbsOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::annotation) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::Annotation>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::assume) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::AssumeOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::bitreverse) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::BitReverseOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::bswap) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::ByteSwapOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_constrained_fptrunc) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands.take_front( 1),
        {}, {}, mlirOperands, mlirAttrs))) {
      return failure();
    }

    FPExceptionBehaviorAttr fpExceptionBehaviorAttr =
        moduleImport.matchFPExceptionBehaviorAttr(llvmOperands[2]);
    mlirAttrs.push_back(
        odsBuilder.getNamedAttr(
            ::mlir::LLVM::ConstrainedFPTruncIntr::getFPExceptionBehaviorAttrName(),
            fpExceptionBehaviorAttr)); 
    RoundingModeAttr roundingModeAttr = moduleImport.matchRoundingModeAttr(llvmOperands[1]);
    mlirAttrs.push_back(
        odsBuilder.getNamedAttr(::mlir::LLVM::ConstrainedFPTruncIntr::getRoundingModeAttrName(),
                               roundingModeAttr));
    
    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::ConstrainedFPTruncIntr>(moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), mlirOperands, mlirAttrs);
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::copysign) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CopySignOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_align) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroAlignOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_begin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroBeginOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_end) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroEndOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_free) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroFreeOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_id) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroIdOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_promise) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroPromiseOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_resume) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::CoroResumeOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_save) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroSaveOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_size) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroSizeOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_suspend) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CoroSuspendOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::cos) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CosOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ctlz) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1},
      {StringLiteral("is_zero_poison")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CountLeadingZerosOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::cttz) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1},
      {StringLiteral("is_zero_poison")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CountTrailingZerosOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ctpop) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::CtPopOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::dbg_declare) {


    // Add debug intrindic to the list of intrinsics that need to be converted once the
    // full function was converted.
    moduleImport.addDebugIntrinsic(inst);
    return success();
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::dbg_label) {


    DILabelAttr labelAttr = moduleImport.matchLabelAttr(llvmOperands[0]);
    // Drop the intrinsic if the label translation fails due to cylic metadata.
    if (!labelAttr)
      return success();
    moduleImport.mapNoResultOp(inst) = odsBuilder.create<::mlir::LLVM::DbgLabelOp>(moduleImport.translateLoc(inst->getDebugLoc()), labelAttr);
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::dbg_value) {


    // Add debug intrindic to the list of intrinsics that need to be converted once the
    // full function was converted.
    moduleImport.addDebugIntrinsic(inst);
    return success();
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::debugtrap) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::DebugTrap>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::eh_typeid_for) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::EhTypeidForOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::exp2) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::Exp2Op>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::exp) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::ExpOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::expect) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::ExpectOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::expect_with_probability) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {2},
      {StringLiteral("prob")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::ExpectWithProbabilityOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::fabs) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FAbsOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ceil) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FCeilOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::floor) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FFloorOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::fma) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FMAOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::fmuladd) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FMulAddOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::trunc) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FTruncOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::fshl) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FshlOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::fshr) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::FshrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::get_active_lane_mask) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::GetActiveLaneMaskOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::invariant_end) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1},
      {StringLiteral("size")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::InvariantEndOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::invariant_start) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {0},
      {StringLiteral("size")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::InvariantStartOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::is_constant) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::IsConstantOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::is_fpclass) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1},
      {StringLiteral("bit")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::IsFPClass>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::lifetime_end) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {0},
      {StringLiteral("size")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::LifetimeEndOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::lifetime_start) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {0},
      {StringLiteral("size")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::LifetimeStartOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::llrint) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::LlrintOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::llround) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::LlroundOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::log2) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::Log2Op>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::log10) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::Log10Op>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::log) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::LogOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::lrint) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::LrintOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::lround) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::LroundOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_load) {
FailureOr<Value> _llvmir_gen_operand_data = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_data))
  return failure();
FailureOr<Value> _llvmir_gen_operand_mask = moduleImport.convertValue(llvmOperands[2]);
if (failed(_llvmir_gen_operand_mask))
  return failure();
FailureOr<SmallVector<Value>> _llvmir_gen_operand_pass_thru = moduleImport.convertValues(llvmOperands.drop_front(3));
if (failed(_llvmir_gen_operand_pass_thru))
  return failure();


    auto *intrinInst = dyn_cast<llvm::IntrinsicInst>(inst);
    bool nontemporal = intrinInst->hasMetadata(llvm::LLVMContext::MD_nontemporal);
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::MaskedLoadOp>(moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_data, *_llvmir_gen_operand_mask, *_llvmir_gen_operand_pass_thru, moduleImport.matchIntegerAttr(llvmOperands[1]),
        nontemporal ? odsBuilder.getUnitAttr() : nullptr);
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_store) {
FailureOr<Value> _llvmir_gen_operand_value = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_value))
  return failure();
FailureOr<Value> _llvmir_gen_operand_data = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_data))
  return failure();
FailureOr<Value> _llvmir_gen_operand_mask = moduleImport.convertValue(llvmOperands[3]);
if (failed(_llvmir_gen_operand_mask))
  return failure();


    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::MaskedStoreOp>(moduleImport.translateLoc(inst->getDebugLoc()),
      *_llvmir_gen_operand_value, *_llvmir_gen_operand_data, *_llvmir_gen_operand_mask, moduleImport.matchIntegerAttr(llvmOperands[2]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::matrix_column_major_load) {
FailureOr<Value> _llvmir_gen_operand_data = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_data))
  return failure();
FailureOr<Value> _llvmir_gen_operand_stride = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_stride))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::MatrixColumnMajorLoadOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_data, *_llvmir_gen_operand_stride,
      moduleImport.matchIntegerAttr(llvmOperands[2]), moduleImport.matchIntegerAttr(llvmOperands[3]), moduleImport.matchIntegerAttr(llvmOperands[4]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::matrix_column_major_store) {
FailureOr<Value> _llvmir_gen_operand_matrix = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_matrix))
  return failure();
FailureOr<Value> _llvmir_gen_operand_data = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_data))
  return failure();
FailureOr<Value> _llvmir_gen_operand_stride = moduleImport.convertValue(llvmOperands[2]);
if (failed(_llvmir_gen_operand_stride))
  return failure();


    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::MatrixColumnMajorStoreOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_matrix, *_llvmir_gen_operand_data, *_llvmir_gen_operand_stride,
      moduleImport.matchIntegerAttr(llvmOperands[3]), moduleImport.matchIntegerAttr(llvmOperands[4]), moduleImport.matchIntegerAttr(llvmOperands[5]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::matrix_multiply) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::MatrixMultiplyOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs,
      moduleImport.matchIntegerAttr(llvmOperands[2]), moduleImport.matchIntegerAttr(llvmOperands[3]), moduleImport.matchIntegerAttr(llvmOperands[4]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::matrix_transpose) {
FailureOr<Value> _llvmir_gen_operand_matrix = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_matrix))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::MatrixTransposeOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_matrix,
      moduleImport.matchIntegerAttr(llvmOperands[1]), moduleImport.matchIntegerAttr(llvmOperands[2]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::maxnum) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::MaxNumOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::maximum) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::MaximumOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memcpy_inline) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {2, 3},
      {StringLiteral("len"), StringLiteral("isVolatile")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::MemcpyInlineOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memcpy) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {3},
      {StringLiteral("isVolatile")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::MemcpyOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memmove) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {3},
      {StringLiteral("isVolatile")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::MemmoveOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memset) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {3},
      {StringLiteral("isVolatile")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::MemsetOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::minnum) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::MinNumOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::minimum) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::MinimumOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nearbyint) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::NearbyintOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_noalias_scope_decl) {


    FailureOr<SmallVector<LLVM::AliasScopeAttr>> scopeAttrs =
      moduleImport.matchAliasScopeAttrs(llvmOperands[0]);
    // Drop the intrinsic if the alias scope translation fails since the scope
    // is not used by an aliasing operation, such as a load or store, that is
    // used to convert the alias scope metadata.
    if (failed(scopeAttrs))
      return success();
    if (scopeAttrs->size() != 1)
      return failure();
    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::NoAliasScopeDeclOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), (*scopeAttrs)[0]);
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::powi) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::PowIOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::pow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::PowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::prefetch) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1, 2, 3},
      {StringLiteral("rw"), StringLiteral("hint"), StringLiteral("cache")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::Prefetch>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ptr_annotation) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::PtrAnnotation>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::rint) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::RintOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::roundeven) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::RoundEvenOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::round) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::RoundOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::sadd_sat) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SAddSat>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::sadd_with_overflow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SAddWithOverflowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::smax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::smin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::smul_with_overflow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SMulWithOverflowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ssa_copy) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SSACopyOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::sshl_sat) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SSHLSat>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ssub_sat) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SSubSat>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ssub_with_overflow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SSubWithOverflowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::sin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::sqrt) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::SqrtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::stackrestore) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::StackRestoreOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::stacksave) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::StackSaveOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::stepvector) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::StepVectorOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::threadlocal_address) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::ThreadlocalAddressOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::trap) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::Trap>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::uadd_sat) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::UAddSat>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::uadd_with_overflow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::UAddWithOverflowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ubsantrap) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {0},
      {StringLiteral("failureKind")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::UBSanTrap>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::umax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::UMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::umin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::UMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::umul_with_overflow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::UMulWithOverflowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::ushl_sat) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::USHLSat>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::usub_sat) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::USubSat>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::usub_with_overflow) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::USubWithOverflowOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_ashr) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPAShrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_add) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPAddOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_and) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPAndOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fadd) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFAddOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fdiv) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFDivOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fmuladd) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFMulAddOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fmul) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFMulOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fneg) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFNegOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fpext) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFPExtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fptosi) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFPToSIOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fptoui) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFPToUIOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fptrunc) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFPTruncOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_frem) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFRemOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fsub) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFSubOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fma) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPFmaOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_inttoptr) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPIntToPtrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_lshr) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPLShrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_load) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPLoadOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_merge) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPMergeMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_mul) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPMulOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_or) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPOrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_ptrtoint) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPPtrToIntOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_add) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceAddOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_and) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceAndOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fadd) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceFAddOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fmax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceFMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fmin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceFMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fmul) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceFMulOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_mul) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceMulOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_or) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceOrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_smax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceSMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_smin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceSMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_umax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceUMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_umin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceUMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_xor) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPReduceXorOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sdiv) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSDivOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sext) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSExtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sitofp) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSIToFPOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_smax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_smin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_srem) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSRemOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_select) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSelectMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_shl) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPShlOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_store) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::VPStoreOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_vp_strided_load) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPStridedLoadOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_vp_strided_store) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::VPStridedStoreOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sub) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPSubOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_trunc) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPTruncOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_udiv) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPUDivOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_uitofp) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPUIToFPOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_umax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPUMaxOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_umin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPUMinOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_urem) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPURemOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_xor) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPXorOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_zext) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::VPZExtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vacopy) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::VaCopyOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vaend) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::VaEndOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vastart) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::VaStartOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::var_annotation) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::VarAnnotation>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_compressstore) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {};
    auto op = odsBuilder.create<::mlir::LLVM::masked_compressstore>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapNoResultOp(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_expandload) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::masked_expandload>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_gather) {
FailureOr<Value> _llvmir_gen_operand_ptrs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_ptrs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_mask = moduleImport.convertValue(llvmOperands[2]);
if (failed(_llvmir_gen_operand_mask))
  return failure();
FailureOr<SmallVector<Value>> _llvmir_gen_operand_pass_thru = moduleImport.convertValues(llvmOperands.drop_front(3));
if (failed(_llvmir_gen_operand_pass_thru))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::masked_gather>(moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_ptrs, *_llvmir_gen_operand_mask, *_llvmir_gen_operand_pass_thru, moduleImport.matchIntegerAttr(llvmOperands[1]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_scatter) {
FailureOr<Value> _llvmir_gen_operand_value = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_value))
  return failure();
FailureOr<Value> _llvmir_gen_operand_ptrs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_ptrs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_mask = moduleImport.convertValue(llvmOperands[3]);
if (failed(_llvmir_gen_operand_mask))
  return failure();


    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::masked_scatter>(moduleImport.translateLoc(inst->getDebugLoc()),
      *_llvmir_gen_operand_value, *_llvmir_gen_operand_ptrs, *_llvmir_gen_operand_mask, moduleImport.matchIntegerAttr(llvmOperands[2]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_deinterleave2) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_deinterleave2>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_extract) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {1},
      {StringLiteral("pos")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_extract>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_insert) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {2},
      {StringLiteral("pos")},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_insert>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_interleave2) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_interleave2>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_add) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_add>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_and) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_and>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fadd) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_fadd>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fmax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_fmax>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fmaximum) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_fmaximum>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fmin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_fmin>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fminimum) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_fminimum>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fmul) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_fmul>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.setFastmathFlagsAttr(inst, op);moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_mul) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_mul>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_or) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_or>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_smax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_smax>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_smin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_smin>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_umax) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_umax>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_umin) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_umin>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_xor) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vector_reduce_xor>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vscale) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
      llvmOperands,
      {},
      {},
      mlirOperands,
      mlirAttrs))
    ) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = odsBuilder.create<::mlir::LLVM::vscale>(
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
