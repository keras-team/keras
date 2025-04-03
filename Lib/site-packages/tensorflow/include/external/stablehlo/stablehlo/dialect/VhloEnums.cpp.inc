/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: VhloEnums.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyComparisonDirectionV1(ComparisonDirectionV1 val) {
  switch (val) {
    case ComparisonDirectionV1::EQ: return "EQ";
    case ComparisonDirectionV1::NE: return "NE";
    case ComparisonDirectionV1::GE: return "GE";
    case ComparisonDirectionV1::GT: return "GT";
    case ComparisonDirectionV1::LE: return "LE";
    case ComparisonDirectionV1::LT: return "LT";
  }
  return "";
}

::std::optional<ComparisonDirectionV1> symbolizeComparisonDirectionV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComparisonDirectionV1>>(str)
      .Case("EQ", ComparisonDirectionV1::EQ)
      .Case("NE", ComparisonDirectionV1::NE)
      .Case("GE", ComparisonDirectionV1::GE)
      .Case("GT", ComparisonDirectionV1::GT)
      .Case("LE", ComparisonDirectionV1::LE)
      .Case("LT", ComparisonDirectionV1::LT)
      .Default(::std::nullopt);
}
::std::optional<ComparisonDirectionV1> symbolizeComparisonDirectionV1(uint32_t value) {
  switch (value) {
  case 0: return ComparisonDirectionV1::EQ;
  case 1: return ComparisonDirectionV1::NE;
  case 2: return ComparisonDirectionV1::GE;
  case 3: return ComparisonDirectionV1::GT;
  case 4: return ComparisonDirectionV1::LE;
  case 5: return ComparisonDirectionV1::LT;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyComparisonTypeV1(ComparisonTypeV1 val) {
  switch (val) {
    case ComparisonTypeV1::NOTYPE: return "NOTYPE";
    case ComparisonTypeV1::FLOAT: return "FLOAT";
    case ComparisonTypeV1::TOTALORDER: return "TOTALORDER";
    case ComparisonTypeV1::SIGNED: return "SIGNED";
    case ComparisonTypeV1::UNSIGNED: return "UNSIGNED";
  }
  return "";
}

::std::optional<ComparisonTypeV1> symbolizeComparisonTypeV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComparisonTypeV1>>(str)
      .Case("NOTYPE", ComparisonTypeV1::NOTYPE)
      .Case("FLOAT", ComparisonTypeV1::FLOAT)
      .Case("TOTALORDER", ComparisonTypeV1::TOTALORDER)
      .Case("SIGNED", ComparisonTypeV1::SIGNED)
      .Case("UNSIGNED", ComparisonTypeV1::UNSIGNED)
      .Default(::std::nullopt);
}
::std::optional<ComparisonTypeV1> symbolizeComparisonTypeV1(uint32_t value) {
  switch (value) {
  case 0: return ComparisonTypeV1::NOTYPE;
  case 1: return ComparisonTypeV1::FLOAT;
  case 2: return ComparisonTypeV1::TOTALORDER;
  case 3: return ComparisonTypeV1::SIGNED;
  case 4: return ComparisonTypeV1::UNSIGNED;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyCustomCallApiVersionV1(CustomCallApiVersionV1 val) {
  switch (val) {
    case CustomCallApiVersionV1::API_VERSION_UNSPECIFIED: return "API_VERSION_UNSPECIFIED";
    case CustomCallApiVersionV1::API_VERSION_ORIGINAL: return "API_VERSION_ORIGINAL";
    case CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING: return "API_VERSION_STATUS_RETURNING";
    case CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING_UNIFIED: return "API_VERSION_STATUS_RETURNING_UNIFIED";
    case CustomCallApiVersionV1::API_VERSION_TYPED_FFI: return "API_VERSION_TYPED_FFI";
  }
  return "";
}

::std::optional<CustomCallApiVersionV1> symbolizeCustomCallApiVersionV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CustomCallApiVersionV1>>(str)
      .Case("API_VERSION_UNSPECIFIED", CustomCallApiVersionV1::API_VERSION_UNSPECIFIED)
      .Case("API_VERSION_ORIGINAL", CustomCallApiVersionV1::API_VERSION_ORIGINAL)
      .Case("API_VERSION_STATUS_RETURNING", CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING)
      .Case("API_VERSION_STATUS_RETURNING_UNIFIED", CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING_UNIFIED)
      .Case("API_VERSION_TYPED_FFI", CustomCallApiVersionV1::API_VERSION_TYPED_FFI)
      .Default(::std::nullopt);
}
::std::optional<CustomCallApiVersionV1> symbolizeCustomCallApiVersionV1(uint32_t value) {
  switch (value) {
  case 0: return CustomCallApiVersionV1::API_VERSION_UNSPECIFIED;
  case 1: return CustomCallApiVersionV1::API_VERSION_ORIGINAL;
  case 2: return CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING;
  case 3: return CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING_UNIFIED;
  case 4: return CustomCallApiVersionV1::API_VERSION_TYPED_FFI;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyFftTypeV1(FftTypeV1 val) {
  switch (val) {
    case FftTypeV1::FFT: return "FFT";
    case FftTypeV1::IFFT: return "IFFT";
    case FftTypeV1::RFFT: return "RFFT";
    case FftTypeV1::IRFFT: return "IRFFT";
  }
  return "";
}

::std::optional<FftTypeV1> symbolizeFftTypeV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FftTypeV1>>(str)
      .Case("FFT", FftTypeV1::FFT)
      .Case("IFFT", FftTypeV1::IFFT)
      .Case("RFFT", FftTypeV1::RFFT)
      .Case("IRFFT", FftTypeV1::IRFFT)
      .Default(::std::nullopt);
}
::std::optional<FftTypeV1> symbolizeFftTypeV1(uint32_t value) {
  switch (value) {
  case 0: return FftTypeV1::FFT;
  case 1: return FftTypeV1::IFFT;
  case 2: return FftTypeV1::RFFT;
  case 3: return FftTypeV1::IRFFT;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyPrecisionV1(PrecisionV1 val) {
  switch (val) {
    case PrecisionV1::DEFAULT: return "DEFAULT";
    case PrecisionV1::HIGH: return "HIGH";
    case PrecisionV1::HIGHEST: return "HIGHEST";
  }
  return "";
}

::std::optional<PrecisionV1> symbolizePrecisionV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<PrecisionV1>>(str)
      .Case("DEFAULT", PrecisionV1::DEFAULT)
      .Case("HIGH", PrecisionV1::HIGH)
      .Case("HIGHEST", PrecisionV1::HIGHEST)
      .Default(::std::nullopt);
}
::std::optional<PrecisionV1> symbolizePrecisionV1(uint32_t value) {
  switch (value) {
  case 0: return PrecisionV1::DEFAULT;
  case 1: return PrecisionV1::HIGH;
  case 2: return PrecisionV1::HIGHEST;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyRngAlgorithmV1(RngAlgorithmV1 val) {
  switch (val) {
    case RngAlgorithmV1::DEFAULT: return "DEFAULT";
    case RngAlgorithmV1::THREE_FRY: return "THREE_FRY";
    case RngAlgorithmV1::PHILOX: return "PHILOX";
  }
  return "";
}

::std::optional<RngAlgorithmV1> symbolizeRngAlgorithmV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RngAlgorithmV1>>(str)
      .Case("DEFAULT", RngAlgorithmV1::DEFAULT)
      .Case("THREE_FRY", RngAlgorithmV1::THREE_FRY)
      .Case("PHILOX", RngAlgorithmV1::PHILOX)
      .Default(::std::nullopt);
}
::std::optional<RngAlgorithmV1> symbolizeRngAlgorithmV1(uint32_t value) {
  switch (value) {
  case 0: return RngAlgorithmV1::DEFAULT;
  case 1: return RngAlgorithmV1::THREE_FRY;
  case 2: return RngAlgorithmV1::PHILOX;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyRngDistributionV1(RngDistributionV1 val) {
  switch (val) {
    case RngDistributionV1::UNIFORM: return "UNIFORM";
    case RngDistributionV1::NORMAL: return "NORMAL";
  }
  return "";
}

::std::optional<RngDistributionV1> symbolizeRngDistributionV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RngDistributionV1>>(str)
      .Case("UNIFORM", RngDistributionV1::UNIFORM)
      .Case("NORMAL", RngDistributionV1::NORMAL)
      .Default(::std::nullopt);
}
::std::optional<RngDistributionV1> symbolizeRngDistributionV1(uint32_t value) {
  switch (value) {
  case 1: return RngDistributionV1::UNIFORM;
  case 2: return RngDistributionV1::NORMAL;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

namespace mlir {
namespace vhlo {
::llvm::StringRef stringifyTransposeV1(TransposeV1 val) {
  switch (val) {
    case TransposeV1::TRANSPOSE_INVALID: return "TRANSPOSE_INVALID";
    case TransposeV1::NO_TRANSPOSE: return "NO_TRANSPOSE";
    case TransposeV1::TRANSPOSE: return "TRANSPOSE";
    case TransposeV1::ADJOINT: return "ADJOINT";
  }
  return "";
}

::std::optional<TransposeV1> symbolizeTransposeV1(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TransposeV1>>(str)
      .Case("TRANSPOSE_INVALID", TransposeV1::TRANSPOSE_INVALID)
      .Case("NO_TRANSPOSE", TransposeV1::NO_TRANSPOSE)
      .Case("TRANSPOSE", TransposeV1::TRANSPOSE)
      .Case("ADJOINT", TransposeV1::ADJOINT)
      .Default(::std::nullopt);
}
::std::optional<TransposeV1> symbolizeTransposeV1(uint32_t value) {
  switch (value) {
  case 0: return TransposeV1::TRANSPOSE_INVALID;
  case 1: return TransposeV1::NO_TRANSPOSE;
  case 2: return TransposeV1::TRANSPOSE;
  case 3: return TransposeV1::ADJOINT;
  default: return ::std::nullopt;
  }
}

} // namespace vhlo
} // namespace mlir

