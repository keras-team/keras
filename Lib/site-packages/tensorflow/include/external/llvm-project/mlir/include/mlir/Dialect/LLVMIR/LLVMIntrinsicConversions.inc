if (auto op = dyn_cast<::mlir::LLVM::AbsOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::abs,1,{},{0},{1},{StringLiteral("is_int_min_poison")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Annotation>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::annotation,1,{0},{2},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AssumeOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::assume,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitReverseOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::bitreverse,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ByteSwapOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::bswap,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ConstrainedFPTruncIntr>(opInst)) {

    SmallVector<llvm::Value *> args =
      moduleTranslation.lookupValues(opInst.getOperands());
    SmallVector<llvm::Type *> overloadedTypes; 
    // Take into account overloaded result type.
    overloadedTypes.push_back(moduleTranslation.convertType(op.getResult().getType())); 
    llvm::transform(ArrayRef<unsigned>{0},
                    std::back_inserter(overloadedTypes),
                    [&args](unsigned index) { return args[index]->getType(); });
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *callee =
      llvm::Intrinsic::getDeclaration(module,
        llvm::Intrinsic::experimental_constrained_fptrunc, overloadedTypes); 
    // Get rounding mode using interface.
    llvm::RoundingMode rounding =
        moduleTranslation.translateRoundingMode(op.getRoundingmode()); 
    llvm::fp::ExceptionBehavior except =
      moduleTranslation.translateFPExceptionBehavior(op.getFpExceptionBehavior());
    moduleTranslation.mapValue(op.getRes()) = builder.CreateConstrainedFPCall(callee, args, "", rounding, except);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CopySignOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::copysign,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroAlignOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_align,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroBeginOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_begin,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroEndOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_end,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroFreeOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_free,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroIdOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_id,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroPromiseOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_promise,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroResumeOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_resume,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSaveOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_save,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSizeOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_size,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CoroSuspendOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::coro_suspend,1,{},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CosOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::cos,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CountLeadingZerosOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ctlz,1,{},{0},{1},{StringLiteral("is_zero_poison")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CountTrailingZerosOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::cttz,1,{},{0},{1},{StringLiteral("is_zero_poison")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CtPopOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ctpop,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::DbgDeclareOp>(opInst)) {

    // Debug intrinsics without debug locations are invalid.
    if(!builder.getCurrentDebugLocation())
      return success();
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::LLVMContext &ctx = module->getContext();
    llvm::Function *fn =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::dbg_declare);
    builder.CreateCall(fn, {
        llvm::MetadataAsValue::get(ctx,
            llvm::ValueAsMetadata::get(moduleTranslation.lookupValue(opInst.getOperand(0)))),
        llvm::MetadataAsValue::get(ctx, moduleTranslation.translateDebugInfo(op.getVarInfo())),
        llvm::MetadataAsValue::get(ctx, moduleTranslation.translateExpression(op.getLocationExpr())),
      });
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::DbgLabelOp>(opInst)) {

    // Debug intrinsics without debug locations are invalid.
    if(!builder.getCurrentDebugLocation())
      return success();
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::LLVMContext &ctx = module->getContext();
    llvm::Function *fn =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::dbg_label);
    builder.CreateCall(fn, {
        llvm::MetadataAsValue::get(ctx, moduleTranslation.translateDebugInfo(op.getLabel()))
      });
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::DbgValueOp>(opInst)) {

    // Debug intrinsics without debug locations are invalid.
    if(!builder.getCurrentDebugLocation())
      return success();
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::LLVMContext &ctx = module->getContext();
    llvm::Function *fn =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::dbg_value);
    builder.CreateCall(fn, {
        llvm::MetadataAsValue::get(ctx,
            llvm::ValueAsMetadata::get(moduleTranslation.lookupValue(opInst.getOperand(0)))),
        llvm::MetadataAsValue::get(ctx, moduleTranslation.translateDebugInfo(op.getVarInfo())),
        llvm::MetadataAsValue::get(ctx, moduleTranslation.translateExpression(op.getLocationExpr())),
      });
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::DebugTrap>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::debugtrap,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::EhTypeidForOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::eh_typeid_for,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Exp2Op>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::exp2,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExpOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::exp,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExpectOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::expect,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExpectWithProbabilityOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::expect_with_probability,1,{},{0},{2},{StringLiteral("prob")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAbsOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::fabs,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCeilOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ceil,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FFloorOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::floor,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMAOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::fma,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulAddOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::fmuladd,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FTruncOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::trunc,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FshlOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::fshl,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FshrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::fshr,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GetActiveLaneMaskOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::get_active_lane_mask,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InvariantEndOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::invariant_end,0,{},{2},{1},{StringLiteral("size")});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InvariantStartOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::invariant_start,1,{},{1},{0},{StringLiteral("size")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::IsConstantOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::is_constant,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::IsFPClass>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::is_fpclass,1,{},{0},{1},{StringLiteral("bit")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LifetimeEndOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::lifetime_end,0,{},{1},{0},{StringLiteral("size")});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LifetimeStartOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::lifetime_start,0,{},{1},{0},{StringLiteral("size")});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LlrintOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::llrint,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LlroundOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::llround,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log2Op>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::log2,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log10Op>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::log10,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LogOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::log,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LrintOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::lrint,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LroundOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::lround,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedLoadOp>(opInst)) {

    auto *inst = moduleTranslation.lookupValues(op.getPassThru()).empty() ? builder.CreateMaskedLoad(
        moduleTranslation.convertType(op.getResult().getType()), moduleTranslation.lookupValue(op.getData()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask())) :
      builder.CreateMaskedLoad(
        moduleTranslation.convertType(op.getResult().getType()), moduleTranslation.lookupValue(op.getData()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()), moduleTranslation.lookupValues(op.getPassThru())[0]);
    moduleTranslation.mapValue(op.getRes()) = inst;
  
    if (op.getNontemporal()) {
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
    }
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedStoreOp>(opInst)) {

    builder.CreateMaskedStore(
      moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getData()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorLoadOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Type *ElemTy = moduleTranslation.convertType(
        getVectorElementType(op.getType()));
    llvm::Align align = dl.getABITypeAlign(ElemTy);
    moduleTranslation.mapValue(op.getRes()) = mb.CreateColumnMajorLoad(
      ElemTy, moduleTranslation.lookupValue(op.getData()), align, moduleTranslation.lookupValue(op.getStride()), op.getIsVolatile(), op.getRows(),
      op.getColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorStoreOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    Type elementType = getVectorElementType(op.getMatrix().getType());
    llvm::Align align = dl.getABITypeAlign(
      moduleTranslation.convertType(elementType));
    mb.CreateColumnMajorStore(
      moduleTranslation.lookupValue(op.getMatrix()), moduleTranslation.lookupValue(op.getData()), align, moduleTranslation.lookupValue(op.getStride()), op.getIsVolatile(),
      op.getRows(), op.getColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixMultiplyOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    moduleTranslation.mapValue(op.getRes()) = mb.CreateMatrixMultiply(
      moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()), op.getLhsRows(), op.getLhsColumns(),
      op.getRhsColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixTransposeOp>(opInst)) {

    llvm::MatrixBuilder mb(builder);
    moduleTranslation.mapValue(op.getRes()) = mb.CreateMatrixTranspose(
      moduleTranslation.lookupValue(op.getMatrix()), op.getRows(), op.getColumns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaxNumOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::maxnum,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaximumOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::maximum,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyInlineOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::memcpy_inline,0,{},{0, 1, 2},{2, 3},{StringLiteral("len"), StringLiteral("isVolatile")});
    (void) inst;
    
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::memcpy,0,{},{0, 1, 2},{3},{StringLiteral("isVolatile")});
    (void) inst;
    
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemmoveOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::memmove,0,{},{0, 1, 2},{3},{StringLiteral("isVolatile")});
    (void) inst;
    
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemsetOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::memset,0,{},{0, 2},{3},{StringLiteral("isVolatile")});
    (void) inst;
    
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinNumOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::minnum,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinimumOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::minimum,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::NearbyintOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::nearbyint,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::NoAliasScopeDeclOp>(opInst)) {

    // Wrap the scope argument into a list since the LLVM IR intrinsic takes
    // a list containing exactly one scope rather than a scope itself.
    llvm::MDNode* node = moduleTranslation.getOrCreateAliasScopes({op.getScope()});
    builder.CreateNoAliasScopeDeclaration(node);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PowIOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::powi,1,{},{0, 1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::pow,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Prefetch>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::prefetch,0,{},{0},{1, 2, 3},{StringLiteral("rw"), StringLiteral("hint"), StringLiteral("cache")});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PtrAnnotation>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ptr_annotation,1,{0},{2},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::RintOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::rint,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::RoundEvenOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::roundeven,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::RoundOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::round,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SAddSat>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::sadd_sat,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SAddWithOverflowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::sadd_with_overflow,2,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::smax,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::smin,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMulWithOverflowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::smul_with_overflow,2,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SSACopyOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ssa_copy,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SSHLSat>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::sshl_sat,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SSubSat>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ssub_sat,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SSubWithOverflowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ssub_with_overflow,2,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::sin,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SqrtOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::sqrt,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StackRestoreOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::stackrestore,0,{},{0},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StackSaveOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::stacksave,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StepVectorOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::stepvector,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ThreadlocalAddressOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::threadlocal_address,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Trap>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::trap,0,{},{},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UAddSat>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::uadd_sat,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UAddWithOverflowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::uadd_with_overflow,2,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UBSanTrap>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ubsantrap,0,{},{},{0},{StringLiteral("failureKind")});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::umax,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::umin,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UMulWithOverflowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::umul_with_overflow,2,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::USHLSat>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::ushl_sat,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::USubSat>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::usub_sat,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::USubWithOverflowOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::usub_with_overflow,2,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPAShrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_ashr,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPAddOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_add,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPAndOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_and,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFAddOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fadd,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFDivOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fdiv,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFMulAddOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fmuladd,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFMulOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fmul,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFNegOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fneg,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFPExtOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fpext,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFPToSIOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fptosi,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFPToUIOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fptoui,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFPTruncOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fptrunc,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFRemOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_frem,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFSubOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fsub,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPFmaOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_fma,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPIntToPtrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_inttoptr,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPLShrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_lshr,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPLoadOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_load,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPMergeMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_merge,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPMulOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_mul,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPOrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_or,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPPtrToIntOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_ptrtoint,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceAddOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_add,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceAndOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_and,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceFAddOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_fadd,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceFMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_fmax,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceFMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_fmin,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceFMulOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_fmul,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceMulOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_mul,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceOrOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_or,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceSMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_smax,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceSMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_smin,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceUMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_umax,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceUMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_umin,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPReduceXorOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_reduce_xor,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSDivOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_sdiv,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSExtOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_sext,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSIToFPOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_sitofp,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_smax,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_smin,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSRemOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_srem,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSelectMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_select,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPShlOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_shl,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPStoreOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_store,0,{},{0, 1},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPStridedLoadOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::experimental_vp_strided_load,1,{0},{0, 1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPStridedStoreOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::experimental_vp_strided_store,0,{},{0, 1, 2},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPSubOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_sub,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPTruncOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_trunc,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPUDivOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_udiv,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPUIToFPOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_uitofp,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPUMaxOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_umax,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPUMinOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_umin,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPURemOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_urem,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPXorOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_xor,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VPZExtOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vp_zext,1,{0},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VaCopyOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vacopy,0,{},{0},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VaEndOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vaend,0,{},{0},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VaStartOp>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vastart,0,{},{0},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VarAnnotation>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::var_annotation,0,{},{0, 1},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_compressstore>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::masked_compressstore,0,{},{0},{},{});
    (void) inst;
    
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_expandload>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::masked_expandload,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_gather>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = moduleTranslation.lookupValues(op.getPassThru()).empty() ? builder.CreateMaskedGather(
        moduleTranslation.convertType(op.getResult().getType()), moduleTranslation.lookupValue(op.getPtrs()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask())) :
      builder.CreateMaskedGather(
        moduleTranslation.convertType(op.getResult().getType()), moduleTranslation.lookupValue(op.getPtrs()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()), moduleTranslation.lookupValues(op.getPassThru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_scatter>(opInst)) {

    builder.CreateMaskedScatter(
      moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getPtrs()), llvm::Align(op.getAlignment()), moduleTranslation.lookupValue(op.getMask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_deinterleave2>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_deinterleave2,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_extract>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_extract,1,{0},{0},{1},{StringLiteral("pos")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_insert>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_insert,1,{0},{1},{2},{StringLiteral("pos")});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_interleave2>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_interleave2,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_add>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_add,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_and>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_and,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fadd>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_fadd,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmax>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_fmax,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmaximum>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_fmaximum,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmin>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_fmin,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fminimum>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_fminimum,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmul>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_fmul,1,{},{1},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_mul>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_mul,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_or>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_or,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smax>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_smax,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smin>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_smin,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umax>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_umax,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umin>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_umin,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_xor>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vector_reduce_xor,1,{},{0},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vscale>(opInst)) {

    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::vscale,1,{0},{},{},{});
    (void) inst;
    moduleTranslation.mapValue(op.getRes()) = inst;
  return success();
}
