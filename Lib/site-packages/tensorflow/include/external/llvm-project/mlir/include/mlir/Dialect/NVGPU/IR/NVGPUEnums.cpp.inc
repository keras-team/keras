/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPU.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace nvgpu {
::llvm::StringRef stringifyRcpRoundingMode(RcpRoundingMode val) {
  switch (val) {
    case RcpRoundingMode::APPROX: return "approx";
    case RcpRoundingMode::RN: return "rn";
    case RcpRoundingMode::RZ: return "rz";
    case RcpRoundingMode::RM: return "rm";
    case RcpRoundingMode::RP: return "rp";
  }
  return "";
}

::std::optional<RcpRoundingMode> symbolizeRcpRoundingMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RcpRoundingMode>>(str)
      .Case("approx", RcpRoundingMode::APPROX)
      .Case("rn", RcpRoundingMode::RN)
      .Case("rz", RcpRoundingMode::RZ)
      .Case("rm", RcpRoundingMode::RM)
      .Case("rp", RcpRoundingMode::RP)
      .Default(::std::nullopt);
}
::std::optional<RcpRoundingMode> symbolizeRcpRoundingMode(uint32_t value) {
  switch (value) {
  case 0: return RcpRoundingMode::APPROX;
  case 1: return RcpRoundingMode::RN;
  case 2: return RcpRoundingMode::RZ;
  case 3: return RcpRoundingMode::RM;
  case 4: return RcpRoundingMode::RP;
  default: return ::std::nullopt;
  }
}

} // namespace nvgpu
} // namespace mlir

namespace mlir {
namespace nvgpu {
::llvm::StringRef stringifyTensorMapInterleaveKind(TensorMapInterleaveKind val) {
  switch (val) {
    case TensorMapInterleaveKind::INTERLEAVE_NONE: return "none";
    case TensorMapInterleaveKind::INTERLEAVE_16B: return "interleave_16b";
    case TensorMapInterleaveKind::INTERLEAVE_32B: return "interleave_32b";
  }
  return "";
}

::std::optional<TensorMapInterleaveKind> symbolizeTensorMapInterleaveKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TensorMapInterleaveKind>>(str)
      .Case("none", TensorMapInterleaveKind::INTERLEAVE_NONE)
      .Case("interleave_16b", TensorMapInterleaveKind::INTERLEAVE_16B)
      .Case("interleave_32b", TensorMapInterleaveKind::INTERLEAVE_32B)
      .Default(::std::nullopt);
}
::std::optional<TensorMapInterleaveKind> symbolizeTensorMapInterleaveKind(uint32_t value) {
  switch (value) {
  case 0: return TensorMapInterleaveKind::INTERLEAVE_NONE;
  case 1: return TensorMapInterleaveKind::INTERLEAVE_16B;
  case 2: return TensorMapInterleaveKind::INTERLEAVE_32B;
  default: return ::std::nullopt;
  }
}

} // namespace nvgpu
} // namespace mlir

namespace mlir {
namespace nvgpu {
::llvm::StringRef stringifyTensorMapL2PromoKind(TensorMapL2PromoKind val) {
  switch (val) {
    case TensorMapL2PromoKind::L2PROMO_NONE: return "none";
    case TensorMapL2PromoKind::L2PROMO_64B: return "l2promo_64b";
    case TensorMapL2PromoKind::L2PROMO_128B: return "l2promo_128b";
    case TensorMapL2PromoKind::L2PROMO_256B: return "l2promo_256b";
  }
  return "";
}

::std::optional<TensorMapL2PromoKind> symbolizeTensorMapL2PromoKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TensorMapL2PromoKind>>(str)
      .Case("none", TensorMapL2PromoKind::L2PROMO_NONE)
      .Case("l2promo_64b", TensorMapL2PromoKind::L2PROMO_64B)
      .Case("l2promo_128b", TensorMapL2PromoKind::L2PROMO_128B)
      .Case("l2promo_256b", TensorMapL2PromoKind::L2PROMO_256B)
      .Default(::std::nullopt);
}
::std::optional<TensorMapL2PromoKind> symbolizeTensorMapL2PromoKind(uint32_t value) {
  switch (value) {
  case 0: return TensorMapL2PromoKind::L2PROMO_NONE;
  case 1: return TensorMapL2PromoKind::L2PROMO_64B;
  case 2: return TensorMapL2PromoKind::L2PROMO_128B;
  case 3: return TensorMapL2PromoKind::L2PROMO_256B;
  default: return ::std::nullopt;
  }
}

} // namespace nvgpu
} // namespace mlir

namespace mlir {
namespace nvgpu {
::llvm::StringRef stringifyTensorMapOOBKind(TensorMapOOBKind val) {
  switch (val) {
    case TensorMapOOBKind::OOB_ZERO: return "zero";
    case TensorMapOOBKind::OOB_NAN: return "nan";
  }
  return "";
}

::std::optional<TensorMapOOBKind> symbolizeTensorMapOOBKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TensorMapOOBKind>>(str)
      .Case("zero", TensorMapOOBKind::OOB_ZERO)
      .Case("nan", TensorMapOOBKind::OOB_NAN)
      .Default(::std::nullopt);
}
::std::optional<TensorMapOOBKind> symbolizeTensorMapOOBKind(uint32_t value) {
  switch (value) {
  case 0: return TensorMapOOBKind::OOB_ZERO;
  case 1: return TensorMapOOBKind::OOB_NAN;
  default: return ::std::nullopt;
  }
}

} // namespace nvgpu
} // namespace mlir

namespace mlir {
namespace nvgpu {
::llvm::StringRef stringifyTensorMapSwizzleKind(TensorMapSwizzleKind val) {
  switch (val) {
    case TensorMapSwizzleKind::SWIZZLE_NONE: return "none";
    case TensorMapSwizzleKind::SWIZZLE_32B: return "swizzle_32b";
    case TensorMapSwizzleKind::SWIZZLE_64B: return "swizzle_64b";
    case TensorMapSwizzleKind::SWIZZLE_128B: return "swizzle_128b";
  }
  return "";
}

::std::optional<TensorMapSwizzleKind> symbolizeTensorMapSwizzleKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TensorMapSwizzleKind>>(str)
      .Case("none", TensorMapSwizzleKind::SWIZZLE_NONE)
      .Case("swizzle_32b", TensorMapSwizzleKind::SWIZZLE_32B)
      .Case("swizzle_64b", TensorMapSwizzleKind::SWIZZLE_64B)
      .Case("swizzle_128b", TensorMapSwizzleKind::SWIZZLE_128B)
      .Default(::std::nullopt);
}
::std::optional<TensorMapSwizzleKind> symbolizeTensorMapSwizzleKind(uint32_t value) {
  switch (value) {
  case 0: return TensorMapSwizzleKind::SWIZZLE_NONE;
  case 1: return TensorMapSwizzleKind::SWIZZLE_32B;
  case 2: return TensorMapSwizzleKind::SWIZZLE_64B;
  case 3: return TensorMapSwizzleKind::SWIZZLE_128B;
  default: return ::std::nullopt;
  }
}

} // namespace nvgpu
} // namespace mlir

