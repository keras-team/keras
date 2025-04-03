static LLVM_ATTRIBUTE_UNUSED ::llvm::InlineAsm::AsmDialect convertAsmDialectToLLVM(::mlir::LLVM::AsmDialect value) {
  switch (value) {
  case ::mlir::LLVM::AsmDialect::AD_ATT:
    return ::llvm::InlineAsm::AsmDialect::AD_ATT;
  case ::mlir::LLVM::AsmDialect::AD_Intel:
    return ::llvm::InlineAsm::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown AsmDialect type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::AtomicRMWInst::BinOp convertAtomicBinOpToLLVM(::mlir::LLVM::AtomicBinOp value) {
  switch (value) {
  case ::mlir::LLVM::AtomicBinOp::xchg:
    return ::llvm::AtomicRMWInst::BinOp::Xchg;
  case ::mlir::LLVM::AtomicBinOp::add:
    return ::llvm::AtomicRMWInst::BinOp::Add;
  case ::mlir::LLVM::AtomicBinOp::sub:
    return ::llvm::AtomicRMWInst::BinOp::Sub;
  case ::mlir::LLVM::AtomicBinOp::_and:
    return ::llvm::AtomicRMWInst::BinOp::And;
  case ::mlir::LLVM::AtomicBinOp::nand:
    return ::llvm::AtomicRMWInst::BinOp::Nand;
  case ::mlir::LLVM::AtomicBinOp::_or:
    return ::llvm::AtomicRMWInst::BinOp::Or;
  case ::mlir::LLVM::AtomicBinOp::_xor:
    return ::llvm::AtomicRMWInst::BinOp::Xor;
  case ::mlir::LLVM::AtomicBinOp::max:
    return ::llvm::AtomicRMWInst::BinOp::Max;
  case ::mlir::LLVM::AtomicBinOp::min:
    return ::llvm::AtomicRMWInst::BinOp::Min;
  case ::mlir::LLVM::AtomicBinOp::umax:
    return ::llvm::AtomicRMWInst::BinOp::UMax;
  case ::mlir::LLVM::AtomicBinOp::umin:
    return ::llvm::AtomicRMWInst::BinOp::UMin;
  case ::mlir::LLVM::AtomicBinOp::fadd:
    return ::llvm::AtomicRMWInst::BinOp::FAdd;
  case ::mlir::LLVM::AtomicBinOp::fsub:
    return ::llvm::AtomicRMWInst::BinOp::FSub;
  case ::mlir::LLVM::AtomicBinOp::fmax:
    return ::llvm::AtomicRMWInst::BinOp::FMax;
  case ::mlir::LLVM::AtomicBinOp::fmin:
    return ::llvm::AtomicRMWInst::BinOp::FMin;
  case ::mlir::LLVM::AtomicBinOp::uinc_wrap:
    return ::llvm::AtomicRMWInst::BinOp::UIncWrap;
  case ::mlir::LLVM::AtomicBinOp::udec_wrap:
    return ::llvm::AtomicRMWInst::BinOp::UDecWrap;
  case ::mlir::LLVM::AtomicBinOp::usub_cond:
    return ::llvm::AtomicRMWInst::BinOp::USubCond;
  case ::mlir::LLVM::AtomicBinOp::usub_sat:
    return ::llvm::AtomicRMWInst::BinOp::USubSat;
  }
  llvm_unreachable("unknown AtomicBinOp type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::AtomicOrdering convertAtomicOrderingToLLVM(::mlir::LLVM::AtomicOrdering value) {
  switch (value) {
  case ::mlir::LLVM::AtomicOrdering::not_atomic:
    return ::llvm::AtomicOrdering::NotAtomic;
  case ::mlir::LLVM::AtomicOrdering::unordered:
    return ::llvm::AtomicOrdering::Unordered;
  case ::mlir::LLVM::AtomicOrdering::monotonic:
    return ::llvm::AtomicOrdering::Monotonic;
  case ::mlir::LLVM::AtomicOrdering::acquire:
    return ::llvm::AtomicOrdering::Acquire;
  case ::mlir::LLVM::AtomicOrdering::release:
    return ::llvm::AtomicOrdering::Release;
  case ::mlir::LLVM::AtomicOrdering::acq_rel:
    return ::llvm::AtomicOrdering::AcquireRelease;
  case ::mlir::LLVM::AtomicOrdering::seq_cst:
    return ::llvm::AtomicOrdering::SequentiallyConsistent;
  }
  llvm_unreachable("unknown AtomicOrdering type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::Comdat::SelectionKind convertComdatToLLVM(::mlir::LLVM::comdat::Comdat value) {
  switch (value) {
  case ::mlir::LLVM::comdat::Comdat::Any:
    return ::llvm::Comdat::SelectionKind::Any;
  case ::mlir::LLVM::comdat::Comdat::ExactMatch:
    return ::llvm::Comdat::SelectionKind::ExactMatch;
  case ::mlir::LLVM::comdat::Comdat::Largest:
    return ::llvm::Comdat::SelectionKind::Largest;
  case ::mlir::LLVM::comdat::Comdat::NoDeduplicate:
    return ::llvm::Comdat::SelectionKind::NoDeduplicate;
  case ::mlir::LLVM::comdat::Comdat::SameSize:
    return ::llvm::Comdat::SelectionKind::SameSize;
  }
  llvm_unreachable("unknown Comdat type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::CmpInst::Predicate convertFCmpPredicateToLLVM(::mlir::LLVM::FCmpPredicate value) {
  switch (value) {
  case ::mlir::LLVM::FCmpPredicate::_false:
    return ::llvm::CmpInst::Predicate::FCMP_FALSE;
  case ::mlir::LLVM::FCmpPredicate::oeq:
    return ::llvm::CmpInst::Predicate::FCMP_OEQ;
  case ::mlir::LLVM::FCmpPredicate::ogt:
    return ::llvm::CmpInst::Predicate::FCMP_OGT;
  case ::mlir::LLVM::FCmpPredicate::oge:
    return ::llvm::CmpInst::Predicate::FCMP_OGE;
  case ::mlir::LLVM::FCmpPredicate::olt:
    return ::llvm::CmpInst::Predicate::FCMP_OLT;
  case ::mlir::LLVM::FCmpPredicate::ole:
    return ::llvm::CmpInst::Predicate::FCMP_OLE;
  case ::mlir::LLVM::FCmpPredicate::one:
    return ::llvm::CmpInst::Predicate::FCMP_ONE;
  case ::mlir::LLVM::FCmpPredicate::ord:
    return ::llvm::CmpInst::Predicate::FCMP_ORD;
  case ::mlir::LLVM::FCmpPredicate::ueq:
    return ::llvm::CmpInst::Predicate::FCMP_UEQ;
  case ::mlir::LLVM::FCmpPredicate::ugt:
    return ::llvm::CmpInst::Predicate::FCMP_UGT;
  case ::mlir::LLVM::FCmpPredicate::uge:
    return ::llvm::CmpInst::Predicate::FCMP_UGE;
  case ::mlir::LLVM::FCmpPredicate::ult:
    return ::llvm::CmpInst::Predicate::FCMP_ULT;
  case ::mlir::LLVM::FCmpPredicate::ule:
    return ::llvm::CmpInst::Predicate::FCMP_ULE;
  case ::mlir::LLVM::FCmpPredicate::une:
    return ::llvm::CmpInst::Predicate::FCMP_UNE;
  case ::mlir::LLVM::FCmpPredicate::uno:
    return ::llvm::CmpInst::Predicate::FCMP_UNO;
  case ::mlir::LLVM::FCmpPredicate::_true:
    return ::llvm::CmpInst::Predicate::FCMP_TRUE;
  }
  llvm_unreachable("unknown FCmpPredicate type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::fp::ExceptionBehavior convertFPExceptionBehaviorToLLVM(::mlir::LLVM::FPExceptionBehavior value) {
  switch (value) {
  case ::mlir::LLVM::FPExceptionBehavior::Ignore:
    return ::llvm::fp::ExceptionBehavior::ebIgnore;
  case ::mlir::LLVM::FPExceptionBehavior::MayTrap:
    return ::llvm::fp::ExceptionBehavior::ebMayTrap;
  case ::mlir::LLVM::FPExceptionBehavior::Strict:
    return ::llvm::fp::ExceptionBehavior::ebStrict;
  }
  llvm_unreachable("unknown FPExceptionBehavior type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::FramePointerKind convertFramePointerKindToLLVM(::mlir::LLVM::framePointerKind::FramePointerKind value) {
  switch (value) {
  case ::mlir::LLVM::framePointerKind::FramePointerKind::None:
    return ::llvm::FramePointerKind::None;
  case ::mlir::LLVM::framePointerKind::FramePointerKind::NonLeaf:
    return ::llvm::FramePointerKind::NonLeaf;
  case ::mlir::LLVM::framePointerKind::FramePointerKind::All:
    return ::llvm::FramePointerKind::All;
  case ::mlir::LLVM::framePointerKind::FramePointerKind::Reserved:
    return ::llvm::FramePointerKind::Reserved;
  }
  llvm_unreachable("unknown FramePointerKind type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::CmpInst::Predicate convertICmpPredicateToLLVM(::mlir::LLVM::ICmpPredicate value) {
  switch (value) {
  case ::mlir::LLVM::ICmpPredicate::eq:
    return ::llvm::CmpInst::Predicate::ICMP_EQ;
  case ::mlir::LLVM::ICmpPredicate::ne:
    return ::llvm::CmpInst::Predicate::ICMP_NE;
  case ::mlir::LLVM::ICmpPredicate::slt:
    return ::llvm::CmpInst::Predicate::ICMP_SLT;
  case ::mlir::LLVM::ICmpPredicate::sle:
    return ::llvm::CmpInst::Predicate::ICMP_SLE;
  case ::mlir::LLVM::ICmpPredicate::sgt:
    return ::llvm::CmpInst::Predicate::ICMP_SGT;
  case ::mlir::LLVM::ICmpPredicate::sge:
    return ::llvm::CmpInst::Predicate::ICMP_SGE;
  case ::mlir::LLVM::ICmpPredicate::ult:
    return ::llvm::CmpInst::Predicate::ICMP_ULT;
  case ::mlir::LLVM::ICmpPredicate::ule:
    return ::llvm::CmpInst::Predicate::ICMP_ULE;
  case ::mlir::LLVM::ICmpPredicate::ugt:
    return ::llvm::CmpInst::Predicate::ICMP_UGT;
  case ::mlir::LLVM::ICmpPredicate::uge:
    return ::llvm::CmpInst::Predicate::ICMP_UGE;
  }
  llvm_unreachable("unknown ICmpPredicate type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::LinkageTypes convertLinkageToLLVM(::mlir::LLVM::linkage::Linkage value) {
  switch (value) {
  case ::mlir::LLVM::linkage::Linkage::Private:
    return ::llvm::GlobalValue::LinkageTypes::PrivateLinkage;
  case ::mlir::LLVM::linkage::Linkage::Internal:
    return ::llvm::GlobalValue::LinkageTypes::InternalLinkage;
  case ::mlir::LLVM::linkage::Linkage::AvailableExternally:
    return ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Linkonce:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Weak:
    return ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage;
  case ::mlir::LLVM::linkage::Linkage::Common:
    return ::llvm::GlobalValue::LinkageTypes::CommonLinkage;
  case ::mlir::LLVM::linkage::Linkage::Appending:
    return ::llvm::GlobalValue::LinkageTypes::AppendingLinkage;
  case ::mlir::LLVM::linkage::Linkage::ExternWeak:
    return ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage;
  case ::mlir::LLVM::linkage::Linkage::LinkonceODR:
    return ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage;
  case ::mlir::LLVM::linkage::Linkage::WeakODR:
    return ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage;
  case ::mlir::LLVM::linkage::Linkage::External:
    return ::llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  }
  llvm_unreachable("unknown Linkage type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::ModRefInfo convertModRefInfoToLLVM(::mlir::LLVM::ModRefInfo value) {
  switch (value) {
  case ::mlir::LLVM::ModRefInfo::NoModRef:
    return ::llvm::ModRefInfo::NoModRef;
  case ::mlir::LLVM::ModRefInfo::Ref:
    return ::llvm::ModRefInfo::Ref;
  case ::mlir::LLVM::ModRefInfo::Mod:
    return ::llvm::ModRefInfo::Mod;
  case ::mlir::LLVM::ModRefInfo::ModRef:
    return ::llvm::ModRefInfo::ModRef;
  }
  llvm_unreachable("unknown ModRefInfo type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::RoundingMode convertRoundingModeToLLVM(::mlir::LLVM::RoundingMode value) {
  switch (value) {
  case ::mlir::LLVM::RoundingMode::TowardZero:
    return ::llvm::RoundingMode::TowardZero;
  case ::mlir::LLVM::RoundingMode::NearestTiesToEven:
    return ::llvm::RoundingMode::NearestTiesToEven;
  case ::mlir::LLVM::RoundingMode::TowardPositive:
    return ::llvm::RoundingMode::TowardPositive;
  case ::mlir::LLVM::RoundingMode::TowardNegative:
    return ::llvm::RoundingMode::TowardNegative;
  case ::mlir::LLVM::RoundingMode::NearestTiesToAway:
    return ::llvm::RoundingMode::NearestTiesToAway;
  case ::mlir::LLVM::RoundingMode::Dynamic:
    return ::llvm::RoundingMode::Dynamic;
  case ::mlir::LLVM::RoundingMode::Invalid:
    return ::llvm::RoundingMode::Invalid;
  }
  llvm_unreachable("unknown RoundingMode type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::CallInst::TailCallKind convertTailCallKindToLLVM(::mlir::LLVM::tailcallkind::TailCallKind value) {
  switch (value) {
  case ::mlir::LLVM::tailcallkind::TailCallKind::None:
    return ::llvm::CallInst::TailCallKind::TCK_None;
  case ::mlir::LLVM::tailcallkind::TailCallKind::NoTail:
    return ::llvm::CallInst::TailCallKind::TCK_NoTail;
  case ::mlir::LLVM::tailcallkind::TailCallKind::MustTail:
    return ::llvm::CallInst::TailCallKind::TCK_MustTail;
  case ::mlir::LLVM::tailcallkind::TailCallKind::Tail:
    return ::llvm::CallInst::TailCallKind::TCK_Tail;
  }
  llvm_unreachable("unknown TailCallKind type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::UnnamedAddr convertUnnamedAddrToLLVM(::mlir::LLVM::UnnamedAddr value) {
  switch (value) {
  case ::mlir::LLVM::UnnamedAddr::None:
    return ::llvm::GlobalValue::UnnamedAddr::None;
  case ::mlir::LLVM::UnnamedAddr::Local:
    return ::llvm::GlobalValue::UnnamedAddr::Local;
  case ::mlir::LLVM::UnnamedAddr::Global:
    return ::llvm::GlobalValue::UnnamedAddr::Global;
  }
  llvm_unreachable("unknown UnnamedAddr type");
}

static LLVM_ATTRIBUTE_UNUSED ::llvm::GlobalValue::VisibilityTypes convertVisibilityToLLVM(::mlir::LLVM::Visibility value) {
  switch (value) {
  case ::mlir::LLVM::Visibility::Default:
    return ::llvm::GlobalValue::VisibilityTypes::DefaultVisibility;
  case ::mlir::LLVM::Visibility::Hidden:
    return ::llvm::GlobalValue::VisibilityTypes::HiddenVisibility;
  case ::mlir::LLVM::Visibility::Protected:
    return ::llvm::GlobalValue::VisibilityTypes::ProtectedVisibility;
  }
  llvm_unreachable("unknown Visibility type");
}

static LLVM_ATTRIBUTE_UNUSED int64_t convertCConvToLLVM(::mlir::LLVM::cconv::CConv value) {
  switch (value) {
  case ::mlir::LLVM::cconv::CConv::C:
    return static_cast<int64_t>(::llvm::CallingConv::C);
  case ::mlir::LLVM::cconv::CConv::Fast:
    return static_cast<int64_t>(::llvm::CallingConv::Fast);
  case ::mlir::LLVM::cconv::CConv::Cold:
    return static_cast<int64_t>(::llvm::CallingConv::Cold);
  case ::mlir::LLVM::cconv::CConv::GHC:
    return static_cast<int64_t>(::llvm::CallingConv::GHC);
  case ::mlir::LLVM::cconv::CConv::HiPE:
    return static_cast<int64_t>(::llvm::CallingConv::HiPE);
  case ::mlir::LLVM::cconv::CConv::AnyReg:
    return static_cast<int64_t>(::llvm::CallingConv::AnyReg);
  case ::mlir::LLVM::cconv::CConv::PreserveMost:
    return static_cast<int64_t>(::llvm::CallingConv::PreserveMost);
  case ::mlir::LLVM::cconv::CConv::PreserveAll:
    return static_cast<int64_t>(::llvm::CallingConv::PreserveAll);
  case ::mlir::LLVM::cconv::CConv::Swift:
    return static_cast<int64_t>(::llvm::CallingConv::Swift);
  case ::mlir::LLVM::cconv::CConv::CXX_FAST_TLS:
    return static_cast<int64_t>(::llvm::CallingConv::CXX_FAST_TLS);
  case ::mlir::LLVM::cconv::CConv::Tail:
    return static_cast<int64_t>(::llvm::CallingConv::Tail);
  case ::mlir::LLVM::cconv::CConv::CFGuard_Check:
    return static_cast<int64_t>(::llvm::CallingConv::CFGuard_Check);
  case ::mlir::LLVM::cconv::CConv::SwiftTail:
    return static_cast<int64_t>(::llvm::CallingConv::SwiftTail);
  case ::mlir::LLVM::cconv::CConv::X86_StdCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_StdCall);
  case ::mlir::LLVM::cconv::CConv::X86_FastCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_FastCall);
  case ::mlir::LLVM::cconv::CConv::ARM_APCS:
    return static_cast<int64_t>(::llvm::CallingConv::ARM_APCS);
  case ::mlir::LLVM::cconv::CConv::ARM_AAPCS:
    return static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS);
  case ::mlir::LLVM::cconv::CConv::ARM_AAPCS_VFP:
    return static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS_VFP);
  case ::mlir::LLVM::cconv::CConv::MSP430_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::MSP430_INTR);
  case ::mlir::LLVM::cconv::CConv::X86_ThisCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_ThisCall);
  case ::mlir::LLVM::cconv::CConv::PTX_Kernel:
    return static_cast<int64_t>(::llvm::CallingConv::PTX_Kernel);
  case ::mlir::LLVM::cconv::CConv::PTX_Device:
    return static_cast<int64_t>(::llvm::CallingConv::PTX_Device);
  case ::mlir::LLVM::cconv::CConv::SPIR_FUNC:
    return static_cast<int64_t>(::llvm::CallingConv::SPIR_FUNC);
  case ::mlir::LLVM::cconv::CConv::SPIR_KERNEL:
    return static_cast<int64_t>(::llvm::CallingConv::SPIR_KERNEL);
  case ::mlir::LLVM::cconv::CConv::Intel_OCL_BI:
    return static_cast<int64_t>(::llvm::CallingConv::Intel_OCL_BI);
  case ::mlir::LLVM::cconv::CConv::X86_64_SysV:
    return static_cast<int64_t>(::llvm::CallingConv::X86_64_SysV);
  case ::mlir::LLVM::cconv::CConv::Win64:
    return static_cast<int64_t>(::llvm::CallingConv::Win64);
  case ::mlir::LLVM::cconv::CConv::X86_VectorCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_VectorCall);
  case ::mlir::LLVM::cconv::CConv::DUMMY_HHVM:
    return static_cast<int64_t>(::llvm::CallingConv::DUMMY_HHVM);
  case ::mlir::LLVM::cconv::CConv::DUMMY_HHVM_C:
    return static_cast<int64_t>(::llvm::CallingConv::DUMMY_HHVM_C);
  case ::mlir::LLVM::cconv::CConv::X86_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::X86_INTR);
  case ::mlir::LLVM::cconv::CConv::AVR_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::AVR_INTR);
  case ::mlir::LLVM::cconv::CConv::AVR_BUILTIN:
    return static_cast<int64_t>(::llvm::CallingConv::AVR_BUILTIN);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_VS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_VS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_GS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_GS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_CS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_CS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_KERNEL:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_KERNEL);
  case ::mlir::LLVM::cconv::CConv::X86_RegCall:
    return static_cast<int64_t>(::llvm::CallingConv::X86_RegCall);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_HS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_HS);
  case ::mlir::LLVM::cconv::CConv::MSP430_BUILTIN:
    return static_cast<int64_t>(::llvm::CallingConv::MSP430_BUILTIN);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_LS:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_LS);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_ES:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_ES);
  case ::mlir::LLVM::cconv::CConv::AArch64_VectorCall:
    return static_cast<int64_t>(::llvm::CallingConv::AArch64_VectorCall);
  case ::mlir::LLVM::cconv::CConv::AArch64_SVE_VectorCall:
    return static_cast<int64_t>(::llvm::CallingConv::AArch64_SVE_VectorCall);
  case ::mlir::LLVM::cconv::CConv::WASM_EmscriptenInvoke:
    return static_cast<int64_t>(::llvm::CallingConv::WASM_EmscriptenInvoke);
  case ::mlir::LLVM::cconv::CConv::AMDGPU_Gfx:
    return static_cast<int64_t>(::llvm::CallingConv::AMDGPU_Gfx);
  case ::mlir::LLVM::cconv::CConv::M68k_INTR:
    return static_cast<int64_t>(::llvm::CallingConv::M68k_INTR);
  }
  llvm_unreachable("unknown CConv type");
}

