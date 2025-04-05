/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: LLVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAsmDialect(AsmDialect val) {
  switch (val) {
    case AsmDialect::AD_ATT: return "att";
    case AsmDialect::AD_Intel: return "intel";
  }
  return "";
}

::std::optional<AsmDialect> symbolizeAsmDialect(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AsmDialect>>(str)
      .Case("att", AsmDialect::AD_ATT)
      .Case("intel", AsmDialect::AD_Intel)
      .Default(::std::nullopt);
}
::std::optional<AsmDialect> symbolizeAsmDialect(uint64_t value) {
  switch (value) {
  case 0: return AsmDialect::AD_ATT;
  case 1: return AsmDialect::AD_Intel;
  default: return ::std::nullopt;
  }
}

bool AsmDialectAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)));
}
AsmDialectAttr AsmDialectAttr::get(::mlir::MLIRContext *context, AsmDialect val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<AsmDialectAttr>(baseAttr);
}
AsmDialect AsmDialectAttr::getValue() const {
  return static_cast<AsmDialect>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp val) {
  switch (val) {
    case AtomicBinOp::xchg: return "xchg";
    case AtomicBinOp::add: return "add";
    case AtomicBinOp::sub: return "sub";
    case AtomicBinOp::_and: return "_and";
    case AtomicBinOp::nand: return "nand";
    case AtomicBinOp::_or: return "_or";
    case AtomicBinOp::_xor: return "_xor";
    case AtomicBinOp::max: return "max";
    case AtomicBinOp::min: return "min";
    case AtomicBinOp::umax: return "umax";
    case AtomicBinOp::umin: return "umin";
    case AtomicBinOp::fadd: return "fadd";
    case AtomicBinOp::fsub: return "fsub";
    case AtomicBinOp::fmax: return "fmax";
    case AtomicBinOp::fmin: return "fmin";
    case AtomicBinOp::uinc_wrap: return "uinc_wrap";
    case AtomicBinOp::udec_wrap: return "udec_wrap";
    case AtomicBinOp::usub_cond: return "usub_cond";
    case AtomicBinOp::usub_sat: return "usub_sat";
  }
  return "";
}

::std::optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AtomicBinOp>>(str)
      .Case("xchg", AtomicBinOp::xchg)
      .Case("add", AtomicBinOp::add)
      .Case("sub", AtomicBinOp::sub)
      .Case("_and", AtomicBinOp::_and)
      .Case("nand", AtomicBinOp::nand)
      .Case("_or", AtomicBinOp::_or)
      .Case("_xor", AtomicBinOp::_xor)
      .Case("max", AtomicBinOp::max)
      .Case("min", AtomicBinOp::min)
      .Case("umax", AtomicBinOp::umax)
      .Case("umin", AtomicBinOp::umin)
      .Case("fadd", AtomicBinOp::fadd)
      .Case("fsub", AtomicBinOp::fsub)
      .Case("fmax", AtomicBinOp::fmax)
      .Case("fmin", AtomicBinOp::fmin)
      .Case("uinc_wrap", AtomicBinOp::uinc_wrap)
      .Case("udec_wrap", AtomicBinOp::udec_wrap)
      .Case("usub_cond", AtomicBinOp::usub_cond)
      .Case("usub_sat", AtomicBinOp::usub_sat)
      .Default(::std::nullopt);
}
::std::optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t value) {
  switch (value) {
  case 0: return AtomicBinOp::xchg;
  case 1: return AtomicBinOp::add;
  case 2: return AtomicBinOp::sub;
  case 3: return AtomicBinOp::_and;
  case 4: return AtomicBinOp::nand;
  case 5: return AtomicBinOp::_or;
  case 6: return AtomicBinOp::_xor;
  case 7: return AtomicBinOp::max;
  case 8: return AtomicBinOp::min;
  case 9: return AtomicBinOp::umax;
  case 10: return AtomicBinOp::umin;
  case 11: return AtomicBinOp::fadd;
  case 12: return AtomicBinOp::fsub;
  case 13: return AtomicBinOp::fmax;
  case 14: return AtomicBinOp::fmin;
  case 15: return AtomicBinOp::uinc_wrap;
  case 16: return AtomicBinOp::udec_wrap;
  case 17: return AtomicBinOp::usub_cond;
  case 18: return AtomicBinOp::usub_sat;
  default: return ::std::nullopt;
  }
}

bool AtomicBinOpAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 11)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 12)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 13)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 14)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 15)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 16)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 17)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 18)));
}
AtomicBinOpAttr AtomicBinOpAttr::get(::mlir::MLIRContext *context, AtomicBinOp val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<AtomicBinOpAttr>(baseAttr);
}
AtomicBinOp AtomicBinOpAttr::getValue() const {
  return static_cast<AtomicBinOp>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering val) {
  switch (val) {
    case AtomicOrdering::not_atomic: return "not_atomic";
    case AtomicOrdering::unordered: return "unordered";
    case AtomicOrdering::monotonic: return "monotonic";
    case AtomicOrdering::acquire: return "acquire";
    case AtomicOrdering::release: return "release";
    case AtomicOrdering::acq_rel: return "acq_rel";
    case AtomicOrdering::seq_cst: return "seq_cst";
  }
  return "";
}

::std::optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AtomicOrdering>>(str)
      .Case("not_atomic", AtomicOrdering::not_atomic)
      .Case("unordered", AtomicOrdering::unordered)
      .Case("monotonic", AtomicOrdering::monotonic)
      .Case("acquire", AtomicOrdering::acquire)
      .Case("release", AtomicOrdering::release)
      .Case("acq_rel", AtomicOrdering::acq_rel)
      .Case("seq_cst", AtomicOrdering::seq_cst)
      .Default(::std::nullopt);
}
::std::optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t value) {
  switch (value) {
  case 0: return AtomicOrdering::not_atomic;
  case 1: return AtomicOrdering::unordered;
  case 2: return AtomicOrdering::monotonic;
  case 4: return AtomicOrdering::acquire;
  case 5: return AtomicOrdering::release;
  case 6: return AtomicOrdering::acq_rel;
  case 7: return AtomicOrdering::seq_cst;
  default: return ::std::nullopt;
  }
}

bool AtomicOrderingAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)));
}
AtomicOrderingAttr AtomicOrderingAttr::get(::mlir::MLIRContext *context, AtomicOrdering val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<AtomicOrderingAttr>(baseAttr);
}
AtomicOrdering AtomicOrderingAttr::getValue() const {
  return static_cast<AtomicOrdering>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace cconv {
::llvm::StringRef stringifyCConv(CConv val) {
  switch (val) {
    case CConv::C: return "ccc";
    case CConv::Fast: return "fastcc";
    case CConv::Cold: return "coldcc";
    case CConv::GHC: return "cc_10";
    case CConv::HiPE: return "cc_11";
    case CConv::AnyReg: return "anyregcc";
    case CConv::PreserveMost: return "preserve_mostcc";
    case CConv::PreserveAll: return "preserve_allcc";
    case CConv::Swift: return "swiftcc";
    case CConv::CXX_FAST_TLS: return "cxx_fast_tlscc";
    case CConv::Tail: return "tailcc";
    case CConv::CFGuard_Check: return "cfguard_checkcc";
    case CConv::SwiftTail: return "swifttailcc";
    case CConv::X86_StdCall: return "x86_stdcallcc";
    case CConv::X86_FastCall: return "x86_fastcallcc";
    case CConv::ARM_APCS: return "arm_apcscc";
    case CConv::ARM_AAPCS: return "arm_aapcscc";
    case CConv::ARM_AAPCS_VFP: return "arm_aapcs_vfpcc";
    case CConv::MSP430_INTR: return "msp430_intrcc";
    case CConv::X86_ThisCall: return "x86_thiscallcc";
    case CConv::PTX_Kernel: return "ptx_kernelcc";
    case CConv::PTX_Device: return "ptx_devicecc";
    case CConv::SPIR_FUNC: return "spir_funccc";
    case CConv::SPIR_KERNEL: return "spir_kernelcc";
    case CConv::Intel_OCL_BI: return "intel_ocl_bicc";
    case CConv::X86_64_SysV: return "x86_64_sysvcc";
    case CConv::Win64: return "win64cc";
    case CConv::X86_VectorCall: return "x86_vectorcallcc";
    case CConv::DUMMY_HHVM: return "hhvmcc";
    case CConv::DUMMY_HHVM_C: return "hhvm_ccc";
    case CConv::X86_INTR: return "x86_intrcc";
    case CConv::AVR_INTR: return "avr_intrcc";
    case CConv::AVR_BUILTIN: return "avr_builtincc";
    case CConv::AMDGPU_VS: return "amdgpu_vscc";
    case CConv::AMDGPU_GS: return "amdgpu_gscc";
    case CConv::AMDGPU_CS: return "amdgpu_cscc";
    case CConv::AMDGPU_KERNEL: return "amdgpu_kernelcc";
    case CConv::X86_RegCall: return "x86_regcallcc";
    case CConv::AMDGPU_HS: return "amdgpu_hscc";
    case CConv::MSP430_BUILTIN: return "msp430_builtincc";
    case CConv::AMDGPU_LS: return "amdgpu_lscc";
    case CConv::AMDGPU_ES: return "amdgpu_escc";
    case CConv::AArch64_VectorCall: return "aarch64_vectorcallcc";
    case CConv::AArch64_SVE_VectorCall: return "aarch64_sve_vectorcallcc";
    case CConv::WASM_EmscriptenInvoke: return "wasm_emscripten_invokecc";
    case CConv::AMDGPU_Gfx: return "amdgpu_gfxcc";
    case CConv::M68k_INTR: return "m68k_intrcc";
  }
  return "";
}

::std::optional<CConv> symbolizeCConv(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CConv>>(str)
      .Case("ccc", CConv::C)
      .Case("fastcc", CConv::Fast)
      .Case("coldcc", CConv::Cold)
      .Case("cc_10", CConv::GHC)
      .Case("cc_11", CConv::HiPE)
      .Case("anyregcc", CConv::AnyReg)
      .Case("preserve_mostcc", CConv::PreserveMost)
      .Case("preserve_allcc", CConv::PreserveAll)
      .Case("swiftcc", CConv::Swift)
      .Case("cxx_fast_tlscc", CConv::CXX_FAST_TLS)
      .Case("tailcc", CConv::Tail)
      .Case("cfguard_checkcc", CConv::CFGuard_Check)
      .Case("swifttailcc", CConv::SwiftTail)
      .Case("x86_stdcallcc", CConv::X86_StdCall)
      .Case("x86_fastcallcc", CConv::X86_FastCall)
      .Case("arm_apcscc", CConv::ARM_APCS)
      .Case("arm_aapcscc", CConv::ARM_AAPCS)
      .Case("arm_aapcs_vfpcc", CConv::ARM_AAPCS_VFP)
      .Case("msp430_intrcc", CConv::MSP430_INTR)
      .Case("x86_thiscallcc", CConv::X86_ThisCall)
      .Case("ptx_kernelcc", CConv::PTX_Kernel)
      .Case("ptx_devicecc", CConv::PTX_Device)
      .Case("spir_funccc", CConv::SPIR_FUNC)
      .Case("spir_kernelcc", CConv::SPIR_KERNEL)
      .Case("intel_ocl_bicc", CConv::Intel_OCL_BI)
      .Case("x86_64_sysvcc", CConv::X86_64_SysV)
      .Case("win64cc", CConv::Win64)
      .Case("x86_vectorcallcc", CConv::X86_VectorCall)
      .Case("hhvmcc", CConv::DUMMY_HHVM)
      .Case("hhvm_ccc", CConv::DUMMY_HHVM_C)
      .Case("x86_intrcc", CConv::X86_INTR)
      .Case("avr_intrcc", CConv::AVR_INTR)
      .Case("avr_builtincc", CConv::AVR_BUILTIN)
      .Case("amdgpu_vscc", CConv::AMDGPU_VS)
      .Case("amdgpu_gscc", CConv::AMDGPU_GS)
      .Case("amdgpu_cscc", CConv::AMDGPU_CS)
      .Case("amdgpu_kernelcc", CConv::AMDGPU_KERNEL)
      .Case("x86_regcallcc", CConv::X86_RegCall)
      .Case("amdgpu_hscc", CConv::AMDGPU_HS)
      .Case("msp430_builtincc", CConv::MSP430_BUILTIN)
      .Case("amdgpu_lscc", CConv::AMDGPU_LS)
      .Case("amdgpu_escc", CConv::AMDGPU_ES)
      .Case("aarch64_vectorcallcc", CConv::AArch64_VectorCall)
      .Case("aarch64_sve_vectorcallcc", CConv::AArch64_SVE_VectorCall)
      .Case("wasm_emscripten_invokecc", CConv::WASM_EmscriptenInvoke)
      .Case("amdgpu_gfxcc", CConv::AMDGPU_Gfx)
      .Case("m68k_intrcc", CConv::M68k_INTR)
      .Default(::std::nullopt);
}
::std::optional<CConv> symbolizeCConv(uint64_t value) {
  switch (value) {
  case 0: return CConv::C;
  case 8: return CConv::Fast;
  case 9: return CConv::Cold;
  case 10: return CConv::GHC;
  case 11: return CConv::HiPE;
  case 13: return CConv::AnyReg;
  case 14: return CConv::PreserveMost;
  case 15: return CConv::PreserveAll;
  case 16: return CConv::Swift;
  case 17: return CConv::CXX_FAST_TLS;
  case 18: return CConv::Tail;
  case 19: return CConv::CFGuard_Check;
  case 20: return CConv::SwiftTail;
  case 64: return CConv::X86_StdCall;
  case 65: return CConv::X86_FastCall;
  case 66: return CConv::ARM_APCS;
  case 67: return CConv::ARM_AAPCS;
  case 68: return CConv::ARM_AAPCS_VFP;
  case 69: return CConv::MSP430_INTR;
  case 70: return CConv::X86_ThisCall;
  case 71: return CConv::PTX_Kernel;
  case 72: return CConv::PTX_Device;
  case 75: return CConv::SPIR_FUNC;
  case 76: return CConv::SPIR_KERNEL;
  case 77: return CConv::Intel_OCL_BI;
  case 78: return CConv::X86_64_SysV;
  case 79: return CConv::Win64;
  case 80: return CConv::X86_VectorCall;
  case 81: return CConv::DUMMY_HHVM;
  case 82: return CConv::DUMMY_HHVM_C;
  case 83: return CConv::X86_INTR;
  case 84: return CConv::AVR_INTR;
  case 86: return CConv::AVR_BUILTIN;
  case 87: return CConv::AMDGPU_VS;
  case 88: return CConv::AMDGPU_GS;
  case 90: return CConv::AMDGPU_CS;
  case 91: return CConv::AMDGPU_KERNEL;
  case 92: return CConv::X86_RegCall;
  case 93: return CConv::AMDGPU_HS;
  case 94: return CConv::MSP430_BUILTIN;
  case 95: return CConv::AMDGPU_LS;
  case 96: return CConv::AMDGPU_ES;
  case 97: return CConv::AArch64_VectorCall;
  case 98: return CConv::AArch64_SVE_VectorCall;
  case 99: return CConv::WASM_EmscriptenInvoke;
  case 100: return CConv::AMDGPU_Gfx;
  case 101: return CConv::M68k_INTR;
  default: return ::std::nullopt;
  }
}

bool CConvAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 11)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 13)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 14)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 15)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 16)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 17)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 18)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 19)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 20)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 64)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 65)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 66)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 67)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 68)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 69)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 70)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 71)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 72)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 75)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 76)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 77)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 78)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 79)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 80)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 81)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 82)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 83)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 84)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 86)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 87)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 88)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 90)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 91)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 92)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 93)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 94)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 95)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 96)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 97)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 98)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 99)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 100)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 101)));
}
CConvAttr CConvAttr::get(::mlir::MLIRContext *context, CConv val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<CConvAttr>(baseAttr);
}
CConv CConvAttr::getValue() const {
  return static_cast<CConv>(::mlir::IntegerAttr::getInt());
}
} // namespace cconv
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace comdat {
::llvm::StringRef stringifyComdat(Comdat val) {
  switch (val) {
    case Comdat::Any: return "any";
    case Comdat::ExactMatch: return "exactmatch";
    case Comdat::Largest: return "largest";
    case Comdat::NoDeduplicate: return "nodeduplicate";
    case Comdat::SameSize: return "samesize";
  }
  return "";
}

::std::optional<Comdat> symbolizeComdat(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Comdat>>(str)
      .Case("any", Comdat::Any)
      .Case("exactmatch", Comdat::ExactMatch)
      .Case("largest", Comdat::Largest)
      .Case("nodeduplicate", Comdat::NoDeduplicate)
      .Case("samesize", Comdat::SameSize)
      .Default(::std::nullopt);
}
::std::optional<Comdat> symbolizeComdat(uint64_t value) {
  switch (value) {
  case 0: return Comdat::Any;
  case 1: return Comdat::ExactMatch;
  case 2: return Comdat::Largest;
  case 3: return Comdat::NoDeduplicate;
  case 4: return Comdat::SameSize;
  default: return ::std::nullopt;
  }
}

bool ComdatAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)));
}
ComdatAttr ComdatAttr::get(::mlir::MLIRContext *context, Comdat val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<ComdatAttr>(baseAttr);
}
Comdat ComdatAttr::getValue() const {
  return static_cast<Comdat>(::mlir::IntegerAttr::getInt());
}
} // namespace comdat
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
std::string stringifyDIFlags(DIFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(1071513599u == (1071513599u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "Zero";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (3u == (3u & val)) {
    strs.push_back("Public");
    val &= ~static_cast<uint32_t>(3);
  }

  if (2u == (2u & val)) {
    strs.push_back("Protected");
    val &= ~static_cast<uint32_t>(2);
  }

  if (1u == (1u & val)) {
    strs.push_back("Private");
    val &= ~static_cast<uint32_t>(1);
  }

  if (1u == (1u & val))
    strs.push_back("Bit0");

  if (2u == (2u & val))
    strs.push_back("Bit1");

  if (4u == (4u & val))
    strs.push_back("FwdDecl");

  if (8u == (8u & val))
    strs.push_back("AppleBlock");

  if (16u == (16u & val))
    strs.push_back("ReservedBit4");

  if (32u == (32u & val))
    strs.push_back("Virtual");

  if (64u == (64u & val))
    strs.push_back("Artificial");

  if (128u == (128u & val))
    strs.push_back("Explicit");

  if (256u == (256u & val))
    strs.push_back("Prototyped");

  if (512u == (512u & val))
    strs.push_back("ObjcClassComplete");

  if (1024u == (1024u & val))
    strs.push_back("ObjectPointer");

  if (2048u == (2048u & val))
    strs.push_back("Vector");

  if (4096u == (4096u & val))
    strs.push_back("StaticMember");

  if (8192u == (8192u & val))
    strs.push_back("LValueReference");

  if (16384u == (16384u & val))
    strs.push_back("RValueReference");

  if (32768u == (32768u & val))
    strs.push_back("ExportSymbols");

  if (65536u == (65536u & val))
    strs.push_back("SingleInheritance");

  if (65536u == (65536u & val))
    strs.push_back("MultipleInheritance");

  if (65536u == (65536u & val))
    strs.push_back("VirtualInheritance");

  if (262144u == (262144u & val))
    strs.push_back("IntroducedVirtual");

  if (524288u == (524288u & val))
    strs.push_back("BitField");

  if (1048576u == (1048576u & val))
    strs.push_back("NoReturn");

  if (4194304u == (4194304u & val))
    strs.push_back("TypePassByValue");

  if (8388608u == (8388608u & val))
    strs.push_back("TypePassByReference");

  if (16777216u == (16777216u & val))
    strs.push_back("EnumClass");

  if (33554432u == (33554432u & val))
    strs.push_back("Thunk");

  if (67108864u == (67108864u & val))
    strs.push_back("NonTrivial");

  if (134217728u == (134217728u & val))
    strs.push_back("BigEndian");

  if (268435456u == (268435456u & val))
    strs.push_back("LittleEndian");

  if (536870912u == (536870912u & val))
    strs.push_back("AllCallsDescribed");
  return ::llvm::join(strs, "|");
}

::std::optional<DIFlags> symbolizeDIFlags(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "Zero") return DIFlags::Zero;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("Bit0", 1)
      .Case("Bit1", 2)
      .Case("Private", 1)
      .Case("Protected", 2)
      .Case("Public", 3)
      .Case("FwdDecl", 4)
      .Case("AppleBlock", 8)
      .Case("ReservedBit4", 16)
      .Case("Virtual", 32)
      .Case("Artificial", 64)
      .Case("Explicit", 128)
      .Case("Prototyped", 256)
      .Case("ObjcClassComplete", 512)
      .Case("ObjectPointer", 1024)
      .Case("Vector", 2048)
      .Case("StaticMember", 4096)
      .Case("LValueReference", 8192)
      .Case("RValueReference", 16384)
      .Case("ExportSymbols", 32768)
      .Case("SingleInheritance", 65536)
      .Case("MultipleInheritance", 65536)
      .Case("VirtualInheritance", 65536)
      .Case("IntroducedVirtual", 262144)
      .Case("BitField", 524288)
      .Case("NoReturn", 1048576)
      .Case("TypePassByValue", 4194304)
      .Case("TypePassByReference", 8388608)
      .Case("EnumClass", 16777216)
      .Case("Thunk", 33554432)
      .Case("NonTrivial", 67108864)
      .Case("BigEndian", 134217728)
      .Case("LittleEndian", 268435456)
      .Case("AllCallsDescribed", 536870912)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<DIFlags>(val);
}

::std::optional<DIFlags> symbolizeDIFlags(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return DIFlags::Zero;

  if (value & ~static_cast<uint32_t>(1071513599u)) return std::nullopt;
  return static_cast<DIFlags>(value);
}
bool DIFlagsAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((!(::llvm::cast<::mlir::IntegerAttr>(attr).getValue().getZExtValue() & (~(0u|1u|2u|1u|2u|3u|4u|8u|16u|32u|64u|128u|256u|512u|1024u|2048u|4096u|8192u|16384u|32768u|65536u|65536u|65536u|262144u|524288u|1048576u|4194304u|8388608u|16777216u|33554432u|67108864u|134217728u|268435456u|536870912u)))));
}
DIFlagsAttr DIFlagsAttr::get(::mlir::MLIRContext *context, DIFlags val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<DIFlagsAttr>(baseAttr);
}
DIFlags DIFlagsAttr::getValue() const {
  return static_cast<DIFlags>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
std::string stringifyDISubprogramFlags(DISubprogramFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(3071u == (3071u | val) && "invalid bits set in bit enum");
  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (1u == (1u & val))
    strs.push_back("Virtual");

  if (2u == (2u & val))
    strs.push_back("PureVirtual");

  if (4u == (4u & val))
    strs.push_back("LocalToUnit");

  if (8u == (8u & val))
    strs.push_back("Definition");

  if (16u == (16u & val))
    strs.push_back("Optimized");

  if (32u == (32u & val))
    strs.push_back("Pure");

  if (64u == (64u & val))
    strs.push_back("Elemental");

  if (128u == (128u & val))
    strs.push_back("Recursive");

  if (256u == (256u & val))
    strs.push_back("MainSubprogram");

  if (512u == (512u & val))
    strs.push_back("Deleted");

  if (2048u == (2048u & val))
    strs.push_back("ObjCDirect");
  return ::llvm::join(strs, "|");
}

::std::optional<DISubprogramFlags> symbolizeDISubprogramFlags(::llvm::StringRef str) {
  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("Virtual", 1)
      .Case("PureVirtual", 2)
      .Case("LocalToUnit", 4)
      .Case("Definition", 8)
      .Case("Optimized", 16)
      .Case("Pure", 32)
      .Case("Elemental", 64)
      .Case("Recursive", 128)
      .Case("MainSubprogram", 256)
      .Case("Deleted", 512)
      .Case("ObjCDirect", 2048)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<DISubprogramFlags>(val);
}

::std::optional<DISubprogramFlags> symbolizeDISubprogramFlags(uint32_t value) {
  if (value & ~static_cast<uint32_t>(3071u)) return std::nullopt;
  return static_cast<DISubprogramFlags>(value);
}
bool DISubprogramFlagsAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((!(::llvm::cast<::mlir::IntegerAttr>(attr).getValue().getZExtValue() & (~(1u|2u|4u|8u|16u|32u|64u|128u|256u|512u|2048u)))));
}
DISubprogramFlagsAttr DISubprogramFlagsAttr::get(::mlir::MLIRContext *context, DISubprogramFlags val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<DISubprogramFlagsAttr>(baseAttr);
}
DISubprogramFlags DISubprogramFlagsAttr::getValue() const {
  return static_cast<DISubprogramFlags>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate val) {
  switch (val) {
    case FCmpPredicate::_false: return "_false";
    case FCmpPredicate::oeq: return "oeq";
    case FCmpPredicate::ogt: return "ogt";
    case FCmpPredicate::oge: return "oge";
    case FCmpPredicate::olt: return "olt";
    case FCmpPredicate::ole: return "ole";
    case FCmpPredicate::one: return "one";
    case FCmpPredicate::ord: return "ord";
    case FCmpPredicate::ueq: return "ueq";
    case FCmpPredicate::ugt: return "ugt";
    case FCmpPredicate::uge: return "uge";
    case FCmpPredicate::ult: return "ult";
    case FCmpPredicate::ule: return "ule";
    case FCmpPredicate::une: return "une";
    case FCmpPredicate::uno: return "uno";
    case FCmpPredicate::_true: return "_true";
  }
  return "";
}

::std::optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FCmpPredicate>>(str)
      .Case("_false", FCmpPredicate::_false)
      .Case("oeq", FCmpPredicate::oeq)
      .Case("ogt", FCmpPredicate::ogt)
      .Case("oge", FCmpPredicate::oge)
      .Case("olt", FCmpPredicate::olt)
      .Case("ole", FCmpPredicate::ole)
      .Case("one", FCmpPredicate::one)
      .Case("ord", FCmpPredicate::ord)
      .Case("ueq", FCmpPredicate::ueq)
      .Case("ugt", FCmpPredicate::ugt)
      .Case("uge", FCmpPredicate::uge)
      .Case("ult", FCmpPredicate::ult)
      .Case("ule", FCmpPredicate::ule)
      .Case("une", FCmpPredicate::une)
      .Case("uno", FCmpPredicate::uno)
      .Case("_true", FCmpPredicate::_true)
      .Default(::std::nullopt);
}
::std::optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return FCmpPredicate::_false;
  case 1: return FCmpPredicate::oeq;
  case 2: return FCmpPredicate::ogt;
  case 3: return FCmpPredicate::oge;
  case 4: return FCmpPredicate::olt;
  case 5: return FCmpPredicate::ole;
  case 6: return FCmpPredicate::one;
  case 7: return FCmpPredicate::ord;
  case 8: return FCmpPredicate::ueq;
  case 9: return FCmpPredicate::ugt;
  case 10: return FCmpPredicate::uge;
  case 11: return FCmpPredicate::ult;
  case 12: return FCmpPredicate::ule;
  case 13: return FCmpPredicate::une;
  case 14: return FCmpPredicate::uno;
  case 15: return FCmpPredicate::_true;
  default: return ::std::nullopt;
  }
}

bool FCmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 11)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 12)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 13)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 14)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 15)));
}
FCmpPredicateAttr FCmpPredicateAttr::get(::mlir::MLIRContext *context, FCmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<FCmpPredicateAttr>(baseAttr);
}
FCmpPredicate FCmpPredicateAttr::getValue() const {
  return static_cast<FCmpPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyFPExceptionBehavior(FPExceptionBehavior val) {
  switch (val) {
    case FPExceptionBehavior::Ignore: return "ignore";
    case FPExceptionBehavior::MayTrap: return "maytrap";
    case FPExceptionBehavior::Strict: return "strict";
  }
  return "";
}

::std::optional<FPExceptionBehavior> symbolizeFPExceptionBehavior(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FPExceptionBehavior>>(str)
      .Case("ignore", FPExceptionBehavior::Ignore)
      .Case("maytrap", FPExceptionBehavior::MayTrap)
      .Case("strict", FPExceptionBehavior::Strict)
      .Default(::std::nullopt);
}
::std::optional<FPExceptionBehavior> symbolizeFPExceptionBehavior(uint64_t value) {
  switch (value) {
  case 0: return FPExceptionBehavior::Ignore;
  case 1: return FPExceptionBehavior::MayTrap;
  case 2: return FPExceptionBehavior::Strict;
  default: return ::std::nullopt;
  }
}

bool FPExceptionBehaviorAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)));
}
FPExceptionBehaviorAttr FPExceptionBehaviorAttr::get(::mlir::MLIRContext *context, FPExceptionBehavior val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<FPExceptionBehaviorAttr>(baseAttr);
}
FPExceptionBehavior FPExceptionBehaviorAttr::getValue() const {
  return static_cast<FPExceptionBehavior>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
std::string stringifyFastmathFlags(FastmathFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(127u == (127u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "none";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (127u == (127u & val)) {
    strs.push_back("fast");
    val &= ~static_cast<uint32_t>(127);
  }

  if (1u == (1u & val))
    strs.push_back("nnan");

  if (2u == (2u & val))
    strs.push_back("ninf");

  if (4u == (4u & val))
    strs.push_back("nsz");

  if (8u == (8u & val))
    strs.push_back("arcp");

  if (16u == (16u & val))
    strs.push_back("contract");

  if (32u == (32u & val))
    strs.push_back("afn");

  if (64u == (64u & val))
    strs.push_back("reassoc");
  return ::llvm::join(strs, ", ");
}

::std::optional<FastmathFlags> symbolizeFastmathFlags(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "none") return FastmathFlags::none;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("nnan", 1)
      .Case("ninf", 2)
      .Case("nsz", 4)
      .Case("arcp", 8)
      .Case("contract", 16)
      .Case("afn", 32)
      .Case("reassoc", 64)
      .Case("fast", 127)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<FastmathFlags>(val);
}

::std::optional<FastmathFlags> symbolizeFastmathFlags(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return FastmathFlags::none;

  if (value & ~static_cast<uint32_t>(127u)) return std::nullopt;
  return static_cast<FastmathFlags>(value);
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace framePointerKind {
::llvm::StringRef stringifyFramePointerKind(FramePointerKind val) {
  switch (val) {
    case FramePointerKind::None: return "none";
    case FramePointerKind::NonLeaf: return "non-leaf";
    case FramePointerKind::All: return "all";
    case FramePointerKind::Reserved: return "reserved";
  }
  return "";
}

::std::optional<FramePointerKind> symbolizeFramePointerKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FramePointerKind>>(str)
      .Case("none", FramePointerKind::None)
      .Case("non-leaf", FramePointerKind::NonLeaf)
      .Case("all", FramePointerKind::All)
      .Case("reserved", FramePointerKind::Reserved)
      .Default(::std::nullopt);
}
::std::optional<FramePointerKind> symbolizeFramePointerKind(uint64_t value) {
  switch (value) {
  case 0: return FramePointerKind::None;
  case 1: return FramePointerKind::NonLeaf;
  case 2: return FramePointerKind::All;
  case 3: return FramePointerKind::Reserved;
  default: return ::std::nullopt;
  }
}

bool FramePointerKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)));
}
FramePointerKindAttr FramePointerKindAttr::get(::mlir::MLIRContext *context, FramePointerKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<FramePointerKindAttr>(baseAttr);
}
FramePointerKind FramePointerKindAttr::getValue() const {
  return static_cast<FramePointerKind>(::mlir::IntegerAttr::getInt());
}
} // namespace framePointerKind
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate val) {
  switch (val) {
    case ICmpPredicate::eq: return "eq";
    case ICmpPredicate::ne: return "ne";
    case ICmpPredicate::slt: return "slt";
    case ICmpPredicate::sle: return "sle";
    case ICmpPredicate::sgt: return "sgt";
    case ICmpPredicate::sge: return "sge";
    case ICmpPredicate::ult: return "ult";
    case ICmpPredicate::ule: return "ule";
    case ICmpPredicate::ugt: return "ugt";
    case ICmpPredicate::uge: return "uge";
  }
  return "";
}

::std::optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ICmpPredicate>>(str)
      .Case("eq", ICmpPredicate::eq)
      .Case("ne", ICmpPredicate::ne)
      .Case("slt", ICmpPredicate::slt)
      .Case("sle", ICmpPredicate::sle)
      .Case("sgt", ICmpPredicate::sgt)
      .Case("sge", ICmpPredicate::sge)
      .Case("ult", ICmpPredicate::ult)
      .Case("ule", ICmpPredicate::ule)
      .Case("ugt", ICmpPredicate::ugt)
      .Case("uge", ICmpPredicate::uge)
      .Default(::std::nullopt);
}
::std::optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return ICmpPredicate::eq;
  case 1: return ICmpPredicate::ne;
  case 2: return ICmpPredicate::slt;
  case 3: return ICmpPredicate::sle;
  case 4: return ICmpPredicate::sgt;
  case 5: return ICmpPredicate::sge;
  case 6: return ICmpPredicate::ult;
  case 7: return ICmpPredicate::ule;
  case 8: return ICmpPredicate::ugt;
  case 9: return ICmpPredicate::uge;
  default: return ::std::nullopt;
  }
}

bool ICmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)));
}
ICmpPredicateAttr ICmpPredicateAttr::get(::mlir::MLIRContext *context, ICmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<ICmpPredicateAttr>(baseAttr);
}
ICmpPredicate ICmpPredicateAttr::getValue() const {
  return static_cast<ICmpPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
std::string stringifyIntegerOverflowFlags(IntegerOverflowFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(3u == (3u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "none";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (1u == (1u & val))
    strs.push_back("nsw");

  if (2u == (2u & val))
    strs.push_back("nuw");
  return ::llvm::join(strs, ", ");
}

::std::optional<IntegerOverflowFlags> symbolizeIntegerOverflowFlags(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "none") return IntegerOverflowFlags::none;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("nsw", 1)
      .Case("nuw", 2)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<IntegerOverflowFlags>(val);
}

::std::optional<IntegerOverflowFlags> symbolizeIntegerOverflowFlags(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return IntegerOverflowFlags::none;

  if (value & ~static_cast<uint32_t>(3u)) return std::nullopt;
  return static_cast<IntegerOverflowFlags>(value);
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyDIEmissionKind(DIEmissionKind val) {
  switch (val) {
    case DIEmissionKind::None: return "None";
    case DIEmissionKind::Full: return "Full";
    case DIEmissionKind::LineTablesOnly: return "LineTablesOnly";
    case DIEmissionKind::DebugDirectivesOnly: return "DebugDirectivesOnly";
  }
  return "";
}

::std::optional<DIEmissionKind> symbolizeDIEmissionKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DIEmissionKind>>(str)
      .Case("None", DIEmissionKind::None)
      .Case("Full", DIEmissionKind::Full)
      .Case("LineTablesOnly", DIEmissionKind::LineTablesOnly)
      .Case("DebugDirectivesOnly", DIEmissionKind::DebugDirectivesOnly)
      .Default(::std::nullopt);
}
::std::optional<DIEmissionKind> symbolizeDIEmissionKind(uint64_t value) {
  switch (value) {
  case 0: return DIEmissionKind::None;
  case 1: return DIEmissionKind::Full;
  case 2: return DIEmissionKind::LineTablesOnly;
  case 3: return DIEmissionKind::DebugDirectivesOnly;
  default: return ::std::nullopt;
  }
}

bool DIEmissionKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)));
}
DIEmissionKindAttr DIEmissionKindAttr::get(::mlir::MLIRContext *context, DIEmissionKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<DIEmissionKindAttr>(baseAttr);
}
DIEmissionKind DIEmissionKindAttr::getValue() const {
  return static_cast<DIEmissionKind>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyDINameTableKind(DINameTableKind val) {
  switch (val) {
    case DINameTableKind::Default: return "Default";
    case DINameTableKind::GNU: return "GNU";
    case DINameTableKind::None: return "None";
    case DINameTableKind::Apple: return "Apple";
  }
  return "";
}

::std::optional<DINameTableKind> symbolizeDINameTableKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DINameTableKind>>(str)
      .Case("Default", DINameTableKind::Default)
      .Case("GNU", DINameTableKind::GNU)
      .Case("None", DINameTableKind::None)
      .Case("Apple", DINameTableKind::Apple)
      .Default(::std::nullopt);
}
::std::optional<DINameTableKind> symbolizeDINameTableKind(uint64_t value) {
  switch (value) {
  case 0: return DINameTableKind::Default;
  case 1: return DINameTableKind::GNU;
  case 2: return DINameTableKind::None;
  case 3: return DINameTableKind::Apple;
  default: return ::std::nullopt;
  }
}

bool DINameTableKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)));
}
DINameTableKindAttr DINameTableKindAttr::get(::mlir::MLIRContext *context, DINameTableKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<DINameTableKindAttr>(baseAttr);
}
DINameTableKind DINameTableKindAttr::getValue() const {
  return static_cast<DINameTableKind>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace linkage {
::llvm::StringRef stringifyLinkage(Linkage val) {
  switch (val) {
    case Linkage::Private: return "private";
    case Linkage::Internal: return "internal";
    case Linkage::AvailableExternally: return "available_externally";
    case Linkage::Linkonce: return "linkonce";
    case Linkage::Weak: return "weak";
    case Linkage::Common: return "common";
    case Linkage::Appending: return "appending";
    case Linkage::ExternWeak: return "extern_weak";
    case Linkage::LinkonceODR: return "linkonce_odr";
    case Linkage::WeakODR: return "weak_odr";
    case Linkage::External: return "external";
  }
  return "";
}

::std::optional<Linkage> symbolizeLinkage(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Linkage>>(str)
      .Case("private", Linkage::Private)
      .Case("internal", Linkage::Internal)
      .Case("available_externally", Linkage::AvailableExternally)
      .Case("linkonce", Linkage::Linkonce)
      .Case("weak", Linkage::Weak)
      .Case("common", Linkage::Common)
      .Case("appending", Linkage::Appending)
      .Case("extern_weak", Linkage::ExternWeak)
      .Case("linkonce_odr", Linkage::LinkonceODR)
      .Case("weak_odr", Linkage::WeakODR)
      .Case("external", Linkage::External)
      .Default(::std::nullopt);
}
::std::optional<Linkage> symbolizeLinkage(uint64_t value) {
  switch (value) {
  case 0: return Linkage::Private;
  case 1: return Linkage::Internal;
  case 2: return Linkage::AvailableExternally;
  case 3: return Linkage::Linkonce;
  case 4: return Linkage::Weak;
  case 5: return Linkage::Common;
  case 6: return Linkage::Appending;
  case 7: return Linkage::ExternWeak;
  case 8: return Linkage::LinkonceODR;
  case 9: return Linkage::WeakODR;
  case 10: return Linkage::External;
  default: return ::std::nullopt;
  }
}

bool LinkageAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)));
}
LinkageAttr LinkageAttr::get(::mlir::MLIRContext *context, Linkage val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<LinkageAttr>(baseAttr);
}
Linkage LinkageAttr::getValue() const {
  return static_cast<Linkage>(::mlir::IntegerAttr::getInt());
}
} // namespace linkage
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyModRefInfo(ModRefInfo val) {
  switch (val) {
    case ModRefInfo::NoModRef: return "none";
    case ModRefInfo::Ref: return "read";
    case ModRefInfo::Mod: return "write";
    case ModRefInfo::ModRef: return "readwrite";
  }
  return "";
}

::std::optional<ModRefInfo> symbolizeModRefInfo(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ModRefInfo>>(str)
      .Case("none", ModRefInfo::NoModRef)
      .Case("read", ModRefInfo::Ref)
      .Case("write", ModRefInfo::Mod)
      .Case("readwrite", ModRefInfo::ModRef)
      .Default(::std::nullopt);
}
::std::optional<ModRefInfo> symbolizeModRefInfo(uint64_t value) {
  switch (value) {
  case 0: return ModRefInfo::NoModRef;
  case 1: return ModRefInfo::Ref;
  case 2: return ModRefInfo::Mod;
  case 3: return ModRefInfo::ModRef;
  default: return ::std::nullopt;
  }
}

bool ModRefInfoAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)));
}
ModRefInfoAttr ModRefInfoAttr::get(::mlir::MLIRContext *context, ModRefInfo val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<ModRefInfoAttr>(baseAttr);
}
ModRefInfo ModRefInfoAttr::getValue() const {
  return static_cast<ModRefInfo>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyRoundingMode(RoundingMode val) {
  switch (val) {
    case RoundingMode::TowardZero: return "towardzero";
    case RoundingMode::NearestTiesToEven: return "tonearest";
    case RoundingMode::TowardPositive: return "upward";
    case RoundingMode::TowardNegative: return "downward";
    case RoundingMode::NearestTiesToAway: return "tonearestaway";
    case RoundingMode::Dynamic: return "dynamic";
    case RoundingMode::Invalid: return "invalid";
  }
  return "";
}

::std::optional<RoundingMode> symbolizeRoundingMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RoundingMode>>(str)
      .Case("towardzero", RoundingMode::TowardZero)
      .Case("tonearest", RoundingMode::NearestTiesToEven)
      .Case("upward", RoundingMode::TowardPositive)
      .Case("downward", RoundingMode::TowardNegative)
      .Case("tonearestaway", RoundingMode::NearestTiesToAway)
      .Case("dynamic", RoundingMode::Dynamic)
      .Case("invalid", RoundingMode::Invalid)
      .Default(::std::nullopt);
}
bool RoundingModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == -1)));
}
RoundingModeAttr RoundingModeAttr::get(::mlir::MLIRContext *context, RoundingMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<RoundingModeAttr>(baseAttr);
}
RoundingMode RoundingModeAttr::getValue() const {
  return static_cast<RoundingMode>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
namespace tailcallkind {
::llvm::StringRef stringifyTailCallKind(TailCallKind val) {
  switch (val) {
    case TailCallKind::None: return "none";
    case TailCallKind::NoTail: return "notail";
    case TailCallKind::MustTail: return "musttail";
    case TailCallKind::Tail: return "tail";
  }
  return "";
}

::std::optional<TailCallKind> symbolizeTailCallKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TailCallKind>>(str)
      .Case("none", TailCallKind::None)
      .Case("notail", TailCallKind::NoTail)
      .Case("musttail", TailCallKind::MustTail)
      .Case("tail", TailCallKind::Tail)
      .Default(::std::nullopt);
}
::std::optional<TailCallKind> symbolizeTailCallKind(uint64_t value) {
  switch (value) {
  case 0: return TailCallKind::None;
  case 3: return TailCallKind::NoTail;
  case 2: return TailCallKind::MustTail;
  case 1: return TailCallKind::Tail;
  default: return ::std::nullopt;
  }
}

bool TailCallKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)));
}
TailCallKindAttr TailCallKindAttr::get(::mlir::MLIRContext *context, TailCallKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<TailCallKindAttr>(baseAttr);
}
TailCallKind TailCallKindAttr::getValue() const {
  return static_cast<TailCallKind>(::mlir::IntegerAttr::getInt());
}
} // namespace tailcallkind
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyUnnamedAddr(UnnamedAddr val) {
  switch (val) {
    case UnnamedAddr::None: return "";
    case UnnamedAddr::Local: return "local_unnamed_addr";
    case UnnamedAddr::Global: return "unnamed_addr";
  }
  return "";
}

::std::optional<UnnamedAddr> symbolizeUnnamedAddr(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<UnnamedAddr>>(str)
      .Case("", UnnamedAddr::None)
      .Case("local_unnamed_addr", UnnamedAddr::Local)
      .Case("unnamed_addr", UnnamedAddr::Global)
      .Default(::std::nullopt);
}
::std::optional<UnnamedAddr> symbolizeUnnamedAddr(uint64_t value) {
  switch (value) {
  case 0: return UnnamedAddr::None;
  case 1: return UnnamedAddr::Local;
  case 2: return UnnamedAddr::Global;
  default: return ::std::nullopt;
  }
}

bool UnnamedAddrAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)));
}
UnnamedAddrAttr UnnamedAddrAttr::get(::mlir::MLIRContext *context, UnnamedAddr val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<UnnamedAddrAttr>(baseAttr);
}
UnnamedAddr UnnamedAddrAttr::getValue() const {
  return static_cast<UnnamedAddr>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyVisibility(Visibility val) {
  switch (val) {
    case Visibility::Default: return "";
    case Visibility::Hidden: return "hidden";
    case Visibility::Protected: return "protected";
  }
  return "";
}

::std::optional<Visibility> symbolizeVisibility(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Visibility>>(str)
      .Case("", Visibility::Default)
      .Case("hidden", Visibility::Hidden)
      .Case("protected", Visibility::Protected)
      .Default(::std::nullopt);
}
::std::optional<Visibility> symbolizeVisibility(uint64_t value) {
  switch (value) {
  case 0: return Visibility::Default;
  case 1: return Visibility::Hidden;
  case 2: return Visibility::Protected;
  default: return ::std::nullopt;
  }
}

bool VisibilityAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)));
}
VisibilityAttr VisibilityAttr::get(::mlir::MLIRContext *context, Visibility val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<VisibilityAttr>(baseAttr);
}
Visibility VisibilityAttr::getValue() const {
  return static_cast<Visibility>(::mlir::IntegerAttr::getInt());
}
} // namespace LLVM
} // namespace mlir

