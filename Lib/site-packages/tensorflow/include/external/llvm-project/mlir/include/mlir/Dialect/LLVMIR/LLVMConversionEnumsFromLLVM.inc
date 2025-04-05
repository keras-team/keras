inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::AsmDialect convertAsmDialectFromLLVM(::llvm::InlineAsm::AsmDialect value) {
  switch (value) {
  case ::llvm::InlineAsm::AsmDialect::AD_ATT:
    return ::mlir::LLVM::AsmDialect::AD_ATT;
  case ::llvm::InlineAsm::AsmDialect::AD_Intel:
    return ::mlir::LLVM::AsmDialect::AD_Intel;
  }
  llvm_unreachable("unknown ::llvm::InlineAsm::AsmDialect type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::AtomicBinOp convertAtomicBinOpFromLLVM(::llvm::AtomicRMWInst::BinOp value) {
  switch (value) {
  case ::llvm::AtomicRMWInst::BinOp::Xchg:
    return ::mlir::LLVM::AtomicBinOp::xchg;
  case ::llvm::AtomicRMWInst::BinOp::Add:
    return ::mlir::LLVM::AtomicBinOp::add;
  case ::llvm::AtomicRMWInst::BinOp::Sub:
    return ::mlir::LLVM::AtomicBinOp::sub;
  case ::llvm::AtomicRMWInst::BinOp::And:
    return ::mlir::LLVM::AtomicBinOp::_and;
  case ::llvm::AtomicRMWInst::BinOp::Nand:
    return ::mlir::LLVM::AtomicBinOp::nand;
  case ::llvm::AtomicRMWInst::BinOp::Or:
    return ::mlir::LLVM::AtomicBinOp::_or;
  case ::llvm::AtomicRMWInst::BinOp::Xor:
    return ::mlir::LLVM::AtomicBinOp::_xor;
  case ::llvm::AtomicRMWInst::BinOp::Max:
    return ::mlir::LLVM::AtomicBinOp::max;
  case ::llvm::AtomicRMWInst::BinOp::Min:
    return ::mlir::LLVM::AtomicBinOp::min;
  case ::llvm::AtomicRMWInst::BinOp::UMax:
    return ::mlir::LLVM::AtomicBinOp::umax;
  case ::llvm::AtomicRMWInst::BinOp::UMin:
    return ::mlir::LLVM::AtomicBinOp::umin;
  case ::llvm::AtomicRMWInst::BinOp::FAdd:
    return ::mlir::LLVM::AtomicBinOp::fadd;
  case ::llvm::AtomicRMWInst::BinOp::FSub:
    return ::mlir::LLVM::AtomicBinOp::fsub;
  case ::llvm::AtomicRMWInst::BinOp::FMax:
    return ::mlir::LLVM::AtomicBinOp::fmax;
  case ::llvm::AtomicRMWInst::BinOp::FMin:
    return ::mlir::LLVM::AtomicBinOp::fmin;
  case ::llvm::AtomicRMWInst::BinOp::UIncWrap:
    return ::mlir::LLVM::AtomicBinOp::uinc_wrap;
  case ::llvm::AtomicRMWInst::BinOp::UDecWrap:
    return ::mlir::LLVM::AtomicBinOp::udec_wrap;
  case ::llvm::AtomicRMWInst::BinOp::USubCond:
    return ::mlir::LLVM::AtomicBinOp::usub_cond;
  case ::llvm::AtomicRMWInst::BinOp::USubSat:
    return ::mlir::LLVM::AtomicBinOp::usub_sat;
  case ::llvm::AtomicRMWInst::BinOp::BAD_BINOP:
    llvm_unreachable("unsupported case ::llvm::AtomicRMWInst::BinOp::BAD_BINOP");
  }
  llvm_unreachable("unknown ::llvm::AtomicRMWInst::BinOp type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::AtomicOrdering convertAtomicOrderingFromLLVM(::llvm::AtomicOrdering value) {
  switch (value) {
  case ::llvm::AtomicOrdering::NotAtomic:
    return ::mlir::LLVM::AtomicOrdering::not_atomic;
  case ::llvm::AtomicOrdering::Unordered:
    return ::mlir::LLVM::AtomicOrdering::unordered;
  case ::llvm::AtomicOrdering::Monotonic:
    return ::mlir::LLVM::AtomicOrdering::monotonic;
  case ::llvm::AtomicOrdering::Acquire:
    return ::mlir::LLVM::AtomicOrdering::acquire;
  case ::llvm::AtomicOrdering::Release:
    return ::mlir::LLVM::AtomicOrdering::release;
  case ::llvm::AtomicOrdering::AcquireRelease:
    return ::mlir::LLVM::AtomicOrdering::acq_rel;
  case ::llvm::AtomicOrdering::SequentiallyConsistent:
    return ::mlir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown ::llvm::AtomicOrdering type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::comdat::Comdat convertComdatFromLLVM(::llvm::Comdat::SelectionKind value) {
  switch (value) {
  case ::llvm::Comdat::SelectionKind::Any:
    return ::mlir::LLVM::comdat::Comdat::Any;
  case ::llvm::Comdat::SelectionKind::ExactMatch:
    return ::mlir::LLVM::comdat::Comdat::ExactMatch;
  case ::llvm::Comdat::SelectionKind::Largest:
    return ::mlir::LLVM::comdat::Comdat::Largest;
  case ::llvm::Comdat::SelectionKind::NoDeduplicate:
    return ::mlir::LLVM::comdat::Comdat::NoDeduplicate;
  case ::llvm::Comdat::SelectionKind::SameSize:
    return ::mlir::LLVM::comdat::Comdat::SameSize;
  }
  llvm_unreachable("unknown ::llvm::Comdat::SelectionKind type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::FCmpPredicate convertFCmpPredicateFromLLVM(::llvm::CmpInst::Predicate value) {
  switch (value) {
  case ::llvm::CmpInst::Predicate::FCMP_FALSE:
    return ::mlir::LLVM::FCmpPredicate::_false;
  case ::llvm::CmpInst::Predicate::FCMP_OEQ:
    return ::mlir::LLVM::FCmpPredicate::oeq;
  case ::llvm::CmpInst::Predicate::FCMP_OGT:
    return ::mlir::LLVM::FCmpPredicate::ogt;
  case ::llvm::CmpInst::Predicate::FCMP_OGE:
    return ::mlir::LLVM::FCmpPredicate::oge;
  case ::llvm::CmpInst::Predicate::FCMP_OLT:
    return ::mlir::LLVM::FCmpPredicate::olt;
  case ::llvm::CmpInst::Predicate::FCMP_OLE:
    return ::mlir::LLVM::FCmpPredicate::ole;
  case ::llvm::CmpInst::Predicate::FCMP_ONE:
    return ::mlir::LLVM::FCmpPredicate::one;
  case ::llvm::CmpInst::Predicate::FCMP_ORD:
    return ::mlir::LLVM::FCmpPredicate::ord;
  case ::llvm::CmpInst::Predicate::FCMP_UEQ:
    return ::mlir::LLVM::FCmpPredicate::ueq;
  case ::llvm::CmpInst::Predicate::FCMP_UGT:
    return ::mlir::LLVM::FCmpPredicate::ugt;
  case ::llvm::CmpInst::Predicate::FCMP_UGE:
    return ::mlir::LLVM::FCmpPredicate::uge;
  case ::llvm::CmpInst::Predicate::FCMP_ULT:
    return ::mlir::LLVM::FCmpPredicate::ult;
  case ::llvm::CmpInst::Predicate::FCMP_ULE:
    return ::mlir::LLVM::FCmpPredicate::ule;
  case ::llvm::CmpInst::Predicate::FCMP_UNE:
    return ::mlir::LLVM::FCmpPredicate::une;
  case ::llvm::CmpInst::Predicate::FCMP_UNO:
    return ::mlir::LLVM::FCmpPredicate::uno;
  case ::llvm::CmpInst::Predicate::FCMP_TRUE:
    return ::mlir::LLVM::FCmpPredicate::_true;
  case ::llvm::CmpInst::Predicate::ICMP_EQ:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_EQ");
  case ::llvm::CmpInst::Predicate::ICMP_NE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_NE");
  case ::llvm::CmpInst::Predicate::ICMP_SLT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_SLT");
  case ::llvm::CmpInst::Predicate::ICMP_SLE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_SLE");
  case ::llvm::CmpInst::Predicate::ICMP_SGT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_SGT");
  case ::llvm::CmpInst::Predicate::ICMP_SGE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_SGE");
  case ::llvm::CmpInst::Predicate::ICMP_ULT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_ULT");
  case ::llvm::CmpInst::Predicate::ICMP_ULE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_ULE");
  case ::llvm::CmpInst::Predicate::ICMP_UGT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_UGT");
  case ::llvm::CmpInst::Predicate::ICMP_UGE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::ICMP_UGE");
  case ::llvm::CmpInst::Predicate::BAD_FCMP_PREDICATE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::BAD_FCMP_PREDICATE");
  case ::llvm::CmpInst::Predicate::BAD_ICMP_PREDICATE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::BAD_ICMP_PREDICATE");
  }
  llvm_unreachable("unknown ::llvm::CmpInst::Predicate type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::FPExceptionBehavior convertFPExceptionBehaviorFromLLVM(::llvm::fp::ExceptionBehavior value) {
  switch (value) {
  case ::llvm::fp::ExceptionBehavior::ebIgnore:
    return ::mlir::LLVM::FPExceptionBehavior::Ignore;
  case ::llvm::fp::ExceptionBehavior::ebMayTrap:
    return ::mlir::LLVM::FPExceptionBehavior::MayTrap;
  case ::llvm::fp::ExceptionBehavior::ebStrict:
    return ::mlir::LLVM::FPExceptionBehavior::Strict;
  }
  llvm_unreachable("unknown ::llvm::fp::ExceptionBehavior type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::framePointerKind::FramePointerKind convertFramePointerKindFromLLVM(::llvm::FramePointerKind value) {
  switch (value) {
  case ::llvm::FramePointerKind::None:
    return ::mlir::LLVM::framePointerKind::FramePointerKind::None;
  case ::llvm::FramePointerKind::NonLeaf:
    return ::mlir::LLVM::framePointerKind::FramePointerKind::NonLeaf;
  case ::llvm::FramePointerKind::All:
    return ::mlir::LLVM::framePointerKind::FramePointerKind::All;
  case ::llvm::FramePointerKind::Reserved:
    return ::mlir::LLVM::framePointerKind::FramePointerKind::Reserved;
  }
  llvm_unreachable("unknown ::llvm::FramePointerKind type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::ICmpPredicate convertICmpPredicateFromLLVM(::llvm::CmpInst::Predicate value) {
  switch (value) {
  case ::llvm::CmpInst::Predicate::ICMP_EQ:
    return ::mlir::LLVM::ICmpPredicate::eq;
  case ::llvm::CmpInst::Predicate::ICMP_NE:
    return ::mlir::LLVM::ICmpPredicate::ne;
  case ::llvm::CmpInst::Predicate::ICMP_SLT:
    return ::mlir::LLVM::ICmpPredicate::slt;
  case ::llvm::CmpInst::Predicate::ICMP_SLE:
    return ::mlir::LLVM::ICmpPredicate::sle;
  case ::llvm::CmpInst::Predicate::ICMP_SGT:
    return ::mlir::LLVM::ICmpPredicate::sgt;
  case ::llvm::CmpInst::Predicate::ICMP_SGE:
    return ::mlir::LLVM::ICmpPredicate::sge;
  case ::llvm::CmpInst::Predicate::ICMP_ULT:
    return ::mlir::LLVM::ICmpPredicate::ult;
  case ::llvm::CmpInst::Predicate::ICMP_ULE:
    return ::mlir::LLVM::ICmpPredicate::ule;
  case ::llvm::CmpInst::Predicate::ICMP_UGT:
    return ::mlir::LLVM::ICmpPredicate::ugt;
  case ::llvm::CmpInst::Predicate::ICMP_UGE:
    return ::mlir::LLVM::ICmpPredicate::uge;
  case ::llvm::CmpInst::Predicate::FCMP_FALSE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_FALSE");
  case ::llvm::CmpInst::Predicate::FCMP_OEQ:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_OEQ");
  case ::llvm::CmpInst::Predicate::FCMP_OGT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_OGT");
  case ::llvm::CmpInst::Predicate::FCMP_OGE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_OGE");
  case ::llvm::CmpInst::Predicate::FCMP_OLT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_OLT");
  case ::llvm::CmpInst::Predicate::FCMP_OLE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_OLE");
  case ::llvm::CmpInst::Predicate::FCMP_ONE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_ONE");
  case ::llvm::CmpInst::Predicate::FCMP_ORD:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_ORD");
  case ::llvm::CmpInst::Predicate::FCMP_UEQ:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_UEQ");
  case ::llvm::CmpInst::Predicate::FCMP_UGT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_UGT");
  case ::llvm::CmpInst::Predicate::FCMP_UGE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_UGE");
  case ::llvm::CmpInst::Predicate::FCMP_ULT:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_ULT");
  case ::llvm::CmpInst::Predicate::FCMP_ULE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_ULE");
  case ::llvm::CmpInst::Predicate::FCMP_UNE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_UNE");
  case ::llvm::CmpInst::Predicate::FCMP_UNO:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_UNO");
  case ::llvm::CmpInst::Predicate::FCMP_TRUE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::FCMP_TRUE");
  case ::llvm::CmpInst::Predicate::BAD_FCMP_PREDICATE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::BAD_FCMP_PREDICATE");
  case ::llvm::CmpInst::Predicate::BAD_ICMP_PREDICATE:
    llvm_unreachable("unsupported case ::llvm::CmpInst::Predicate::BAD_ICMP_PREDICATE");
  }
  llvm_unreachable("unknown ::llvm::CmpInst::Predicate type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::linkage::Linkage convertLinkageFromLLVM(::llvm::GlobalValue::LinkageTypes value) {
  switch (value) {
  case ::llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    return ::mlir::LLVM::linkage::Linkage::Private;
  case ::llvm::GlobalValue::LinkageTypes::InternalLinkage:
    return ::mlir::LLVM::linkage::Linkage::Internal;
  case ::llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    return ::mlir::LLVM::linkage::Linkage::AvailableExternally;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    return ::mlir::LLVM::linkage::Linkage::Linkonce;
  case ::llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    return ::mlir::LLVM::linkage::Linkage::Weak;
  case ::llvm::GlobalValue::LinkageTypes::CommonLinkage:
    return ::mlir::LLVM::linkage::Linkage::Common;
  case ::llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    return ::mlir::LLVM::linkage::Linkage::Appending;
  case ::llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    return ::mlir::LLVM::linkage::Linkage::ExternWeak;
  case ::llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    return ::mlir::LLVM::linkage::Linkage::LinkonceODR;
  case ::llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    return ::mlir::LLVM::linkage::Linkage::WeakODR;
  case ::llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    return ::mlir::LLVM::linkage::Linkage::External;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::LinkageTypes type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::ModRefInfo convertModRefInfoFromLLVM(::llvm::ModRefInfo value) {
  switch (value) {
  case ::llvm::ModRefInfo::NoModRef:
    return ::mlir::LLVM::ModRefInfo::NoModRef;
  case ::llvm::ModRefInfo::Ref:
    return ::mlir::LLVM::ModRefInfo::Ref;
  case ::llvm::ModRefInfo::Mod:
    return ::mlir::LLVM::ModRefInfo::Mod;
  case ::llvm::ModRefInfo::ModRef:
    return ::mlir::LLVM::ModRefInfo::ModRef;
  }
  llvm_unreachable("unknown ::llvm::ModRefInfo type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::RoundingMode convertRoundingModeFromLLVM(::llvm::RoundingMode value) {
  switch (value) {
  case ::llvm::RoundingMode::TowardZero:
    return ::mlir::LLVM::RoundingMode::TowardZero;
  case ::llvm::RoundingMode::NearestTiesToEven:
    return ::mlir::LLVM::RoundingMode::NearestTiesToEven;
  case ::llvm::RoundingMode::TowardPositive:
    return ::mlir::LLVM::RoundingMode::TowardPositive;
  case ::llvm::RoundingMode::TowardNegative:
    return ::mlir::LLVM::RoundingMode::TowardNegative;
  case ::llvm::RoundingMode::NearestTiesToAway:
    return ::mlir::LLVM::RoundingMode::NearestTiesToAway;
  case ::llvm::RoundingMode::Dynamic:
    return ::mlir::LLVM::RoundingMode::Dynamic;
  case ::llvm::RoundingMode::Invalid:
    return ::mlir::LLVM::RoundingMode::Invalid;
  }
  llvm_unreachable("unknown ::llvm::RoundingMode type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::tailcallkind::TailCallKind convertTailCallKindFromLLVM(::llvm::CallInst::TailCallKind value) {
  switch (value) {
  case ::llvm::CallInst::TailCallKind::TCK_None:
    return ::mlir::LLVM::tailcallkind::TailCallKind::None;
  case ::llvm::CallInst::TailCallKind::TCK_NoTail:
    return ::mlir::LLVM::tailcallkind::TailCallKind::NoTail;
  case ::llvm::CallInst::TailCallKind::TCK_MustTail:
    return ::mlir::LLVM::tailcallkind::TailCallKind::MustTail;
  case ::llvm::CallInst::TailCallKind::TCK_Tail:
    return ::mlir::LLVM::tailcallkind::TailCallKind::Tail;
  }
  llvm_unreachable("unknown ::llvm::CallInst::TailCallKind type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::UnnamedAddr convertUnnamedAddrFromLLVM(::llvm::GlobalValue::UnnamedAddr value) {
  switch (value) {
  case ::llvm::GlobalValue::UnnamedAddr::None:
    return ::mlir::LLVM::UnnamedAddr::None;
  case ::llvm::GlobalValue::UnnamedAddr::Local:
    return ::mlir::LLVM::UnnamedAddr::Local;
  case ::llvm::GlobalValue::UnnamedAddr::Global:
    return ::mlir::LLVM::UnnamedAddr::Global;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::UnnamedAddr type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::Visibility convertVisibilityFromLLVM(::llvm::GlobalValue::VisibilityTypes value) {
  switch (value) {
  case ::llvm::GlobalValue::VisibilityTypes::DefaultVisibility:
    return ::mlir::LLVM::Visibility::Default;
  case ::llvm::GlobalValue::VisibilityTypes::HiddenVisibility:
    return ::mlir::LLVM::Visibility::Hidden;
  case ::llvm::GlobalValue::VisibilityTypes::ProtectedVisibility:
    return ::mlir::LLVM::Visibility::Protected;
  }
  llvm_unreachable("unknown ::llvm::GlobalValue::VisibilityTypes type");}

inline LLVM_ATTRIBUTE_UNUSED ::mlir::LLVM::cconv::CConv convertCConvFromLLVM(int64_t value) {
  switch (value) {
  case static_cast<int64_t>(::llvm::CallingConv::C):
    return ::mlir::LLVM::cconv::CConv::C;
  case static_cast<int64_t>(::llvm::CallingConv::Fast):
    return ::mlir::LLVM::cconv::CConv::Fast;
  case static_cast<int64_t>(::llvm::CallingConv::Cold):
    return ::mlir::LLVM::cconv::CConv::Cold;
  case static_cast<int64_t>(::llvm::CallingConv::GHC):
    return ::mlir::LLVM::cconv::CConv::GHC;
  case static_cast<int64_t>(::llvm::CallingConv::HiPE):
    return ::mlir::LLVM::cconv::CConv::HiPE;
  case static_cast<int64_t>(::llvm::CallingConv::AnyReg):
    return ::mlir::LLVM::cconv::CConv::AnyReg;
  case static_cast<int64_t>(::llvm::CallingConv::PreserveMost):
    return ::mlir::LLVM::cconv::CConv::PreserveMost;
  case static_cast<int64_t>(::llvm::CallingConv::PreserveAll):
    return ::mlir::LLVM::cconv::CConv::PreserveAll;
  case static_cast<int64_t>(::llvm::CallingConv::Swift):
    return ::mlir::LLVM::cconv::CConv::Swift;
  case static_cast<int64_t>(::llvm::CallingConv::CXX_FAST_TLS):
    return ::mlir::LLVM::cconv::CConv::CXX_FAST_TLS;
  case static_cast<int64_t>(::llvm::CallingConv::Tail):
    return ::mlir::LLVM::cconv::CConv::Tail;
  case static_cast<int64_t>(::llvm::CallingConv::CFGuard_Check):
    return ::mlir::LLVM::cconv::CConv::CFGuard_Check;
  case static_cast<int64_t>(::llvm::CallingConv::SwiftTail):
    return ::mlir::LLVM::cconv::CConv::SwiftTail;
  case static_cast<int64_t>(::llvm::CallingConv::X86_StdCall):
    return ::mlir::LLVM::cconv::CConv::X86_StdCall;
  case static_cast<int64_t>(::llvm::CallingConv::X86_FastCall):
    return ::mlir::LLVM::cconv::CConv::X86_FastCall;
  case static_cast<int64_t>(::llvm::CallingConv::ARM_APCS):
    return ::mlir::LLVM::cconv::CConv::ARM_APCS;
  case static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS):
    return ::mlir::LLVM::cconv::CConv::ARM_AAPCS;
  case static_cast<int64_t>(::llvm::CallingConv::ARM_AAPCS_VFP):
    return ::mlir::LLVM::cconv::CConv::ARM_AAPCS_VFP;
  case static_cast<int64_t>(::llvm::CallingConv::MSP430_INTR):
    return ::mlir::LLVM::cconv::CConv::MSP430_INTR;
  case static_cast<int64_t>(::llvm::CallingConv::X86_ThisCall):
    return ::mlir::LLVM::cconv::CConv::X86_ThisCall;
  case static_cast<int64_t>(::llvm::CallingConv::PTX_Kernel):
    return ::mlir::LLVM::cconv::CConv::PTX_Kernel;
  case static_cast<int64_t>(::llvm::CallingConv::PTX_Device):
    return ::mlir::LLVM::cconv::CConv::PTX_Device;
  case static_cast<int64_t>(::llvm::CallingConv::SPIR_FUNC):
    return ::mlir::LLVM::cconv::CConv::SPIR_FUNC;
  case static_cast<int64_t>(::llvm::CallingConv::SPIR_KERNEL):
    return ::mlir::LLVM::cconv::CConv::SPIR_KERNEL;
  case static_cast<int64_t>(::llvm::CallingConv::Intel_OCL_BI):
    return ::mlir::LLVM::cconv::CConv::Intel_OCL_BI;
  case static_cast<int64_t>(::llvm::CallingConv::X86_64_SysV):
    return ::mlir::LLVM::cconv::CConv::X86_64_SysV;
  case static_cast<int64_t>(::llvm::CallingConv::Win64):
    return ::mlir::LLVM::cconv::CConv::Win64;
  case static_cast<int64_t>(::llvm::CallingConv::X86_VectorCall):
    return ::mlir::LLVM::cconv::CConv::X86_VectorCall;
  case static_cast<int64_t>(::llvm::CallingConv::DUMMY_HHVM):
    return ::mlir::LLVM::cconv::CConv::DUMMY_HHVM;
  case static_cast<int64_t>(::llvm::CallingConv::DUMMY_HHVM_C):
    return ::mlir::LLVM::cconv::CConv::DUMMY_HHVM_C;
  case static_cast<int64_t>(::llvm::CallingConv::X86_INTR):
    return ::mlir::LLVM::cconv::CConv::X86_INTR;
  case static_cast<int64_t>(::llvm::CallingConv::AVR_INTR):
    return ::mlir::LLVM::cconv::CConv::AVR_INTR;
  case static_cast<int64_t>(::llvm::CallingConv::AVR_BUILTIN):
    return ::mlir::LLVM::cconv::CConv::AVR_BUILTIN;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_VS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_VS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_GS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_GS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_CS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_CS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_KERNEL):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_KERNEL;
  case static_cast<int64_t>(::llvm::CallingConv::X86_RegCall):
    return ::mlir::LLVM::cconv::CConv::X86_RegCall;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_HS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_HS;
  case static_cast<int64_t>(::llvm::CallingConv::MSP430_BUILTIN):
    return ::mlir::LLVM::cconv::CConv::MSP430_BUILTIN;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_LS):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_LS;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_ES):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_ES;
  case static_cast<int64_t>(::llvm::CallingConv::AArch64_VectorCall):
    return ::mlir::LLVM::cconv::CConv::AArch64_VectorCall;
  case static_cast<int64_t>(::llvm::CallingConv::AArch64_SVE_VectorCall):
    return ::mlir::LLVM::cconv::CConv::AArch64_SVE_VectorCall;
  case static_cast<int64_t>(::llvm::CallingConv::WASM_EmscriptenInvoke):
    return ::mlir::LLVM::cconv::CConv::WASM_EmscriptenInvoke;
  case static_cast<int64_t>(::llvm::CallingConv::AMDGPU_Gfx):
    return ::mlir::LLVM::cconv::CConv::AMDGPU_Gfx;
  case static_cast<int64_t>(::llvm::CallingConv::M68k_INTR):
    return ::mlir::LLVM::cconv::CConv::M68k_INTR;
  }
  llvm_unreachable("unknown ::llvm::CallingConv type");}

