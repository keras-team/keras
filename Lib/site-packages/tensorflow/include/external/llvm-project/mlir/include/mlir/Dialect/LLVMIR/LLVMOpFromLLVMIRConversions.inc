if (inst->getOpcode() == llvm::Instruction::AShr) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::AShrOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Add) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::AddOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setIntegerOverflowFlags(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::AddrSpaceCast) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::AddrSpaceCastOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Alloca) {
FailureOr<Value> _llvmir_gen_operand_arraySize = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arraySize))
  return failure();


    auto *allocaInst = cast<llvm::AllocaInst>(inst);
    Type allocatedType =
      moduleImport.convertType(allocaInst->getAllocatedType());
    unsigned alignment = allocaInst->getAlign().value();
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::AllocaOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arraySize,
      alignment == 0 ? IntegerAttr() : odsBuilder.getI64IntegerAttr(alignment),
      allocatedType, allocaInst->isUsedWithInAlloca());
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::And) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::AndOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::AtomicCmpXchg) {
FailureOr<Value> _llvmir_gen_operand_ptr = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_ptr))
  return failure();
FailureOr<Value> _llvmir_gen_operand_cmp = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_cmp))
  return failure();
FailureOr<Value> _llvmir_gen_operand_val = moduleImport.convertValue(llvmOperands[2]);
if (failed(_llvmir_gen_operand_val))
  return failure();


    auto *cmpXchgInst = cast<llvm::AtomicCmpXchgInst>(inst);
    unsigned alignment = cmpXchgInst->getAlign().value();
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::AtomicCmpXchgOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_ptr, *_llvmir_gen_operand_cmp, *_llvmir_gen_operand_val,
      convertAtomicOrderingFromLLVM(cmpXchgInst->getSuccessOrdering()),
      convertAtomicOrderingFromLLVM(cmpXchgInst->getFailureOrdering()),
      getLLVMSyncScope(cmpXchgInst), alignment, cmpXchgInst->isWeak(),
      cmpXchgInst->isVolatile());
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::AtomicRMW) {
FailureOr<Value> _llvmir_gen_operand_ptr = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_ptr))
  return failure();
FailureOr<Value> _llvmir_gen_operand_val = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_val))
  return failure();


    auto *atomicInst = cast<llvm::AtomicRMWInst>(inst);
    unsigned alignment = atomicInst->getAlign().value();
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::AtomicRMWOp>(moduleImport.translateLoc(inst->getDebugLoc()),
        convertAtomicBinOpFromLLVM(atomicInst->getOperation()), *_llvmir_gen_operand_ptr, *_llvmir_gen_operand_val,
        convertAtomicOrderingFromLLVM(atomicInst->getOrdering()),
        getLLVMSyncScope(atomicInst), alignment, atomicInst->isVolatile());
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::BitCast) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::BitcastOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ExtractElement) {
FailureOr<Value> _llvmir_gen_operand_vector = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_vector))
  return failure();
FailureOr<Value> _llvmir_gen_operand_position = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_position))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::ExtractElementOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_vector, *_llvmir_gen_operand_position);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ExtractValue) {
FailureOr<Value> _llvmir_gen_operand_container = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_container))
  return failure();


    auto *evInst = cast<llvm::ExtractValueInst>(inst);
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::ExtractValueOp>(moduleImport.translateLoc(inst->getDebugLoc()),
      *_llvmir_gen_operand_container, getPositionFromIndices(evInst->getIndices()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FAdd) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::FAddOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FCmp) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto *fCmpInst = cast<llvm::FCmpInst>(inst);
    auto op = odsBuilder.create<::mlir::LLVM::FCmpOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), convertFCmpPredicateFromLLVM(fCmpInst->getPredicate()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FDiv) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::FDivOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FMul) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::FMulOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FNeg) {
FailureOr<Value> _llvmir_gen_operand_operand = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_operand))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::FNegOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_operand);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
   
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPExt) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPExtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPToSI) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPToSIOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPToUI) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPToUIOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPTrunc) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPTruncOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FRem) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::FRemOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FSub) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::FSubOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Fence) {


    llvm::FenceInst *fenceInst = cast<llvm::FenceInst>(inst);
    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::FenceOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      convertAtomicOrderingFromLLVM(fenceInst->getOrdering()),
      getLLVMSyncScope(fenceInst));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Freeze) {
FailureOr<Value> _llvmir_gen_operand_val = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_val))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::FreezeOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_val);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ICmp) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto *iCmpInst = cast<llvm::ICmpInst>(inst);
    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::ICmpOp>(moduleImport.translateLoc(inst->getDebugLoc()),
            convertICmpPredicateFromLLVM(iCmpInst->getPredicate()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::InsertElement) {
FailureOr<Value> _llvmir_gen_operand_vector = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_vector))
  return failure();
FailureOr<Value> _llvmir_gen_operand_value = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_value))
  return failure();
FailureOr<Value> _llvmir_gen_operand_position = moduleImport.convertValue(llvmOperands[2]);
if (failed(_llvmir_gen_operand_position))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::InsertElementOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_vector, *_llvmir_gen_operand_value, *_llvmir_gen_operand_position);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::InsertValue) {
FailureOr<Value> _llvmir_gen_operand_container = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_container))
  return failure();
FailureOr<Value> _llvmir_gen_operand_value = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_value))
  return failure();


    auto *ivInst = cast<llvm::InsertValueInst>(inst);
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::InsertValueOp>(moduleImport.translateLoc(inst->getDebugLoc()),
      *_llvmir_gen_operand_container, *_llvmir_gen_operand_value, getPositionFromIndices(ivInst->getIndices()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::IntToPtr) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::IntToPtrOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::LShr) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::LShrOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Load) {
FailureOr<Value> _llvmir_gen_operand_addr = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_addr))
  return failure();


    auto *loadInst = cast<llvm::LoadInst>(inst);
    unsigned alignment = loadInst->getAlign().value();
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::LoadOp>(moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_addr,
        alignment, loadInst->isVolatile(),
        loadInst->hasMetadata(llvm::LLVMContext::MD_nontemporal),
        loadInst->hasMetadata(llvm::LLVMContext::MD_invariant_load),
        convertAtomicOrderingFromLLVM(loadInst->getOrdering()),
        getLLVMSyncScope(loadInst));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Mul) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::MulOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setIntegerOverflowFlags(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Or) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::OrOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::PtrToInt) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::PtrToIntOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Resume) {
FailureOr<Value> _llvmir_gen_operand_value = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_value))
  return failure();


    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::ResumeOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_value);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Ret) {


    FailureOr<SmallVector<Value>> mlirOperands =
      moduleImport.convertValues(llvmOperands);
    if (failed(mlirOperands))
      return failure();
    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::ReturnOp>(moduleImport.translateLoc(inst->getDebugLoc()), *mlirOperands);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SDiv) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::SDivOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SExt) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::SExtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SIToFP) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::SIToFPOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SRem) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::SRemOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Select) {
FailureOr<Value> _llvmir_gen_operand_condition = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_condition))
  return failure();
FailureOr<Value> _llvmir_gen_operand_trueValue = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_trueValue))
  return failure();
FailureOr<Value> _llvmir_gen_operand_falseValue = moduleImport.convertValue(llvmOperands[2]);
if (failed(_llvmir_gen_operand_falseValue))
  return failure();


    auto op = odsBuilder.create<LLVM::SelectOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_condition, *_llvmir_gen_operand_trueValue, *_llvmir_gen_operand_falseValue);
    moduleImport.setFastmathFlagsAttr(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Shl) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::ShlOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setIntegerOverflowFlags(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ShuffleVector) {
FailureOr<Value> _llvmir_gen_operand_v1 = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_v1))
  return failure();
FailureOr<Value> _llvmir_gen_operand_v2 = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_v2))
  return failure();


    auto *svInst = cast<llvm::ShuffleVectorInst>(inst);
    SmallVector<int32_t> mask(svInst->getShuffleMask());
    moduleImport.mapValue(inst) = odsBuilder.create<LLVM::ShuffleVectorOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_v1, *_llvmir_gen_operand_v2, mask);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Store) {
FailureOr<Value> _llvmir_gen_operand_value = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_value))
  return failure();
FailureOr<Value> _llvmir_gen_operand_addr = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_addr))
  return failure();


    auto *storeInst = cast<llvm::StoreInst>(inst);
    unsigned alignment = storeInst->getAlign().value();
    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::StoreOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_value, *_llvmir_gen_operand_addr,
        alignment, storeInst->isVolatile(),
        storeInst->hasMetadata(llvm::LLVMContext::MD_nontemporal),
        convertAtomicOrderingFromLLVM(storeInst->getOrdering()),
        getLLVMSyncScope(storeInst));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Sub) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    auto op = odsBuilder.create<::mlir::LLVM::SubOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
    moduleImport.setIntegerOverflowFlags(inst, op);
    moduleImport.mapValue(inst) = op;
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Trunc) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::TruncOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::UDiv) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::UDivOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::UIToFP) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::UIToFPOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::URem) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::URemOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Unreachable) {


    moduleImport.mapNoResultOp(inst) = odsBuilder.create<LLVM::UnreachableOp>(moduleImport.translateLoc(inst->getDebugLoc()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::VAArg) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<mlir::LLVM::VaArgOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Xor) {
FailureOr<Value> _llvmir_gen_operand_lhs = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_lhs))
  return failure();
FailureOr<Value> _llvmir_gen_operand_rhs = moduleImport.convertValue(llvmOperands[1]);
if (failed(_llvmir_gen_operand_rhs))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::XOrOp>(moduleImport.translateLoc(inst->getDebugLoc()), *_llvmir_gen_operand_lhs, *_llvmir_gen_operand_rhs);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ZExt) {
FailureOr<Value> _llvmir_gen_operand_arg = moduleImport.convertValue(llvmOperands[0]);
if (failed(_llvmir_gen_operand_arg))
  return failure();


    moduleImport.mapValue(inst) = odsBuilder.create<::mlir::LLVM::ZExtOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), moduleImport.convertType(inst->getType()), *_llvmir_gen_operand_arg);
  
  return success();
}
