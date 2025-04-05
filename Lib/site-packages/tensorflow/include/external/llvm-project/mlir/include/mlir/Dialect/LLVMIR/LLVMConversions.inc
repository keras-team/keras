if (auto op = dyn_cast<::mlir::LLVM::AShrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAShr(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAdd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()), /*Name=*/"", op.hasNoUnsignedWrap(), op.hasNoSignedWrap());
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddrSpaceCastOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAddrSpaceCast(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AllocaOp>(opInst)) {

    auto addrSpace = moduleTranslation.convertType(op.getResult().getType())->getPointerAddressSpace();
    llvm::Type *elementType = moduleTranslation.convertType(op.getElemType());
    auto *inst = builder.CreateAlloca(elementType, addrSpace, moduleTranslation.lookupValue(op.getArraySize()));
    
    if (op.getAlignment().has_value()) {
      auto align = *op.getAlignment();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    inst->setUsedWithInAlloca(op.getInalloca());
    moduleTranslation.mapValue(op.getRes()) = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AndOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAnd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicCmpXchgOp>(opInst)) {

    auto *inst = builder.CreateAtomicCmpXchg(moduleTranslation.lookupValue(op.getPtr()), moduleTranslation.lookupValue(op.getCmp()), moduleTranslation.lookupValue(op.getVal()),
        llvm::MaybeAlign(), convertAtomicOrderingToLLVM(op.getSuccessOrdering()),
        convertAtomicOrderingToLLVM(op.getFailureOrdering()));
    moduleTranslation.mapValue(op.getRes()) = inst;
    inst->setWeak(op.getWeak());
  
    inst->setVolatile(op.getVolatile_());
  
    if (op.getSyncscope().has_value()) {
      llvm::LLVMContext &llvmContext = builder.getContext();
      inst->setSyncScopeID(llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
    }
  
    if (op.getAlignment().has_value()) {
      auto align = *op.getAlignment();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicRMWOp>(opInst)) {

    auto *inst = builder.CreateAtomicRMW(
        convertAtomicBinOpToLLVM(op.getBinOp()), moduleTranslation.lookupValue(op.getPtr()), moduleTranslation.lookupValue(op.getVal()), llvm::MaybeAlign(),
        convertAtomicOrderingToLLVM(op.getOrdering()));
    moduleTranslation.mapValue(op.getRes()) = inst;
  
    inst->setVolatile(op.getVolatile_());
  
    if (op.getSyncscope().has_value()) {
      llvm::LLVMContext &llvmContext = builder.getContext();
      inst->setSyncScopeID(llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
    }
  
    if (op.getAlignment().has_value()) {
      auto align = *op.getAlignment();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitcastOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateBitCast(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CallIntrinsicOp>(opInst)) {

    return convertCallLLVMIntrinsicOp(op, builder, moduleTranslation);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ConstantOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = getLLVMConstant(moduleTranslation.convertType(op.getResult().getType()), op.getValue(), opInst.getLoc(),
                                            moduleTranslation);
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractElementOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateExtractElement(moduleTranslation.lookupValue(op.getVector()), moduleTranslation.lookupValue(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractValueOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateExtractValue(moduleTranslation.lookupValue(op.getContainer()), extractPosition(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAddOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFAdd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCmpOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateFCmp(convertFCmpPredicateToLLVM(op.getPredicate()), moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FDivOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFDiv(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFMul(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FNegOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFNeg(moduleTranslation.lookupValue(op.getOperand()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPExtOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPExt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToSIOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPToSI(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToUIOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPToUI(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPTruncOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPTrunc(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FRemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFRem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FSubOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFSub(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FenceOp>(opInst)) {

    auto *inst = builder.CreateFence(convertAtomicOrderingToLLVM(op.getOrdering()));
  
    if (op.getSyncscope().has_value()) {
      llvm::LLVMContext &llvmContext = builder.getContext();
      inst->setSyncScopeID(llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
    }
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FreezeOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFreeze(moduleTranslation.lookupValue(op.getVal()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GEPOp>(opInst)) {

    SmallVector<llvm::Value *> indices;
    indices.reserve(op.getRawConstantIndices().size());
    GEPIndicesAdaptor<decltype(moduleTranslation.lookupValues(op.getDynamicIndices()))>
        gepIndices(op.getRawConstantIndicesAttr(), moduleTranslation.lookupValues(op.getDynamicIndices()));
    for (PointerUnion<IntegerAttr, llvm::Value*> valueOrAttr : gepIndices) {
      if (llvm::Value* value = ::llvm::dyn_cast<llvm::Value*>(valueOrAttr))
        indices.push_back(value);
      else
        indices.push_back(
            builder.getInt32(valueOrAttr.get<IntegerAttr>().getInt()));
    }
    Type baseElementType = op.getElemType();
    llvm::Type *elementType = moduleTranslation.convertType(baseElementType);
    moduleTranslation.mapValue(op.getRes()) = builder.CreateGEP(elementType, moduleTranslation.lookupValue(op.getBase()), indices, "", op.getInbounds());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ICmpOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateICmp(
            convertICmpPredicateToLLVM(op.getPredicate()), moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertElementOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateInsertElement(moduleTranslation.lookupValue(op.getVector()), moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertValueOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateInsertValue(moduleTranslation.lookupValue(op.getContainer()), moduleTranslation.lookupValue(op.getValue()),
                                     extractPosition(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::IntToPtrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateIntToPtr(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LShrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateLShr(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LinkerOptionsOp>(opInst)) {

    convertLinkerOptionsOp(op.getOptions(), builder, moduleTranslation);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LoadOp>(opInst)) {

    auto *inst = builder.CreateLoad(moduleTranslation.convertType(op.getResult().getType()), moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
    moduleTranslation.mapValue(op.getRes()) = inst;
    if (op.getInvariant()) {
      llvm::MDNode *metadata = llvm::MDNode::get(inst->getContext(), std::nullopt);
      inst->setMetadata(llvm::LLVMContext::MD_invariant_load, metadata);
    }
  
    inst->setAtomic(convertAtomicOrderingToLLVM(op.getOrdering()));
  
    if (op.getSyncscope().has_value()) {
      llvm::LLVMContext &llvmContext = builder.getContext();
      inst->setSyncScopeID(llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
    }
  
    if (op.getAlignment().has_value()) {
      auto align = *op.getAlignment();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.getNontemporal()) {
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MulOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateMul(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()), /*Name=*/"", op.hasNoUnsignedWrap(), op.hasNoSignedWrap());
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::NoneTokenOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = llvm::ConstantTokenNone::get(builder.getContext());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::OrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateOr(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PoisonOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = llvm::PoisonValue::get(moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PtrToIntOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreatePtrToInt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ResumeOp>(opInst)) {
 builder.CreateResume(moduleTranslation.lookupValue(op.getValue())); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ReturnOp>(opInst)) {

    if (opInst.getNumOperands() != 0)
      builder.CreateRet(moduleTranslation.lookupValue(op.getArg()));
    else
      builder.CreateRetVoid();
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SDivOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSDiv(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SExtOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSExt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SIToFPOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSIToFP(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SRemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSRem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SelectOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSelect(moduleTranslation.lookupValue(op.getCondition()), moduleTranslation.lookupValue(op.getTrueValue()), moduleTranslation.lookupValue(op.getFalseValue()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShlOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateShl(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()), /*Name=*/"", op.hasNoUnsignedWrap(), op.hasNoSignedWrap());
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShuffleVectorOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateShuffleVector(moduleTranslation.lookupValue(op.getV1()), moduleTranslation.lookupValue(op.getV2()), op.getMask());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StoreOp>(opInst)) {

    auto *inst = builder.CreateStore(moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
  
    inst->setAtomic(convertAtomicOrderingToLLVM(op.getOrdering()));
  
    if (op.getSyncscope().has_value()) {
      llvm::LLVMContext &llvmContext = builder.getContext();
      inst->setSyncScopeID(llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
    }
  
    if (op.getAlignment().has_value()) {
      auto align = *op.getAlignment();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.getNontemporal()) {
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SubOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSub(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()), /*Name=*/"", op.hasNoUnsignedWrap(), op.hasNoSignedWrap());
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::TruncOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateTrunc(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UDivOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateUDiv(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UIToFPOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateUIToFP(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::URemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateURem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UndefOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = llvm::UndefValue::get(moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UnreachableOp>(opInst)) {
 builder.CreateUnreachable(); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::VaArgOp>(opInst)) {
 moduleTranslation.mapValue(op.getRes()) = builder.CreateVAArg(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType())); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::XOrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateXor(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ZExtOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateZExt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ZeroOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = llvm::Constant::getNullValue(moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
