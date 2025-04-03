/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: StablehloOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyComparisonDirection(ComparisonDirection val) {
  switch (val) {
    case ComparisonDirection::EQ: return "EQ";
    case ComparisonDirection::NE: return "NE";
    case ComparisonDirection::GE: return "GE";
    case ComparisonDirection::GT: return "GT";
    case ComparisonDirection::LE: return "LE";
    case ComparisonDirection::LT: return "LT";
  }
  return "";
}

::std::optional<ComparisonDirection> symbolizeComparisonDirection(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComparisonDirection>>(str)
      .Case("EQ", ComparisonDirection::EQ)
      .Case("NE", ComparisonDirection::NE)
      .Case("GE", ComparisonDirection::GE)
      .Case("GT", ComparisonDirection::GT)
      .Case("LE", ComparisonDirection::LE)
      .Case("LT", ComparisonDirection::LT)
      .Default(::std::nullopt);
}
::std::optional<ComparisonDirection> symbolizeComparisonDirection(uint32_t value) {
  switch (value) {
  case 0: return ComparisonDirection::EQ;
  case 1: return ComparisonDirection::NE;
  case 2: return ComparisonDirection::GE;
  case 3: return ComparisonDirection::GT;
  case 4: return ComparisonDirection::LE;
  case 5: return ComparisonDirection::LT;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyComparisonType(ComparisonType val) {
  switch (val) {
    case ComparisonType::NOTYPE: return "NOTYPE";
    case ComparisonType::FLOAT: return "FLOAT";
    case ComparisonType::TOTALORDER: return "TOTALORDER";
    case ComparisonType::SIGNED: return "SIGNED";
    case ComparisonType::UNSIGNED: return "UNSIGNED";
  }
  return "";
}

::std::optional<ComparisonType> symbolizeComparisonType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComparisonType>>(str)
      .Case("NOTYPE", ComparisonType::NOTYPE)
      .Case("FLOAT", ComparisonType::FLOAT)
      .Case("TOTALORDER", ComparisonType::TOTALORDER)
      .Case("SIGNED", ComparisonType::SIGNED)
      .Case("UNSIGNED", ComparisonType::UNSIGNED)
      .Default(::std::nullopt);
}
::std::optional<ComparisonType> symbolizeComparisonType(uint32_t value) {
  switch (value) {
  case 0: return ComparisonType::NOTYPE;
  case 1: return ComparisonType::FLOAT;
  case 2: return ComparisonType::TOTALORDER;
  case 3: return ComparisonType::SIGNED;
  case 4: return ComparisonType::UNSIGNED;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyCustomCallApiVersion(CustomCallApiVersion val) {
  switch (val) {
    case CustomCallApiVersion::API_VERSION_UNSPECIFIED: return "API_VERSION_UNSPECIFIED";
    case CustomCallApiVersion::API_VERSION_ORIGINAL: return "API_VERSION_ORIGINAL";
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING: return "API_VERSION_STATUS_RETURNING";
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED: return "API_VERSION_STATUS_RETURNING_UNIFIED";
    case CustomCallApiVersion::API_VERSION_TYPED_FFI: return "API_VERSION_TYPED_FFI";
  }
  return "";
}

::std::optional<CustomCallApiVersion> symbolizeCustomCallApiVersion(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CustomCallApiVersion>>(str)
      .Case("API_VERSION_UNSPECIFIED", CustomCallApiVersion::API_VERSION_UNSPECIFIED)
      .Case("API_VERSION_ORIGINAL", CustomCallApiVersion::API_VERSION_ORIGINAL)
      .Case("API_VERSION_STATUS_RETURNING", CustomCallApiVersion::API_VERSION_STATUS_RETURNING)
      .Case("API_VERSION_STATUS_RETURNING_UNIFIED", CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED)
      .Case("API_VERSION_TYPED_FFI", CustomCallApiVersion::API_VERSION_TYPED_FFI)
      .Default(::std::nullopt);
}
::std::optional<CustomCallApiVersion> symbolizeCustomCallApiVersion(uint32_t value) {
  switch (value) {
  case 0: return CustomCallApiVersion::API_VERSION_UNSPECIFIED;
  case 1: return CustomCallApiVersion::API_VERSION_ORIGINAL;
  case 2: return CustomCallApiVersion::API_VERSION_STATUS_RETURNING;
  case 3: return CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED;
  case 4: return CustomCallApiVersion::API_VERSION_TYPED_FFI;
  default: return ::std::nullopt;
  }
}

bool CustomCallApiVersionAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)));
}
CustomCallApiVersionAttr CustomCallApiVersionAttr::get(::mlir::MLIRContext *context, CustomCallApiVersion val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<CustomCallApiVersionAttr>(baseAttr);
}
CustomCallApiVersion CustomCallApiVersionAttr::getValue() const {
  return static_cast<CustomCallApiVersion>(::mlir::IntegerAttr::getInt());
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyFftType(FftType val) {
  switch (val) {
    case FftType::FFT: return "FFT";
    case FftType::IFFT: return "IFFT";
    case FftType::RFFT: return "RFFT";
    case FftType::IRFFT: return "IRFFT";
  }
  return "";
}

::std::optional<FftType> symbolizeFftType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FftType>>(str)
      .Case("FFT", FftType::FFT)
      .Case("IFFT", FftType::IFFT)
      .Case("RFFT", FftType::RFFT)
      .Case("IRFFT", FftType::IRFFT)
      .Default(::std::nullopt);
}
::std::optional<FftType> symbolizeFftType(uint32_t value) {
  switch (value) {
  case 0: return FftType::FFT;
  case 1: return FftType::IFFT;
  case 2: return FftType::RFFT;
  case 3: return FftType::IRFFT;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyPrecision(Precision val) {
  switch (val) {
    case Precision::DEFAULT: return "DEFAULT";
    case Precision::HIGH: return "HIGH";
    case Precision::HIGHEST: return "HIGHEST";
  }
  return "";
}

::std::optional<Precision> symbolizePrecision(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Precision>>(str)
      .Case("DEFAULT", Precision::DEFAULT)
      .Case("HIGH", Precision::HIGH)
      .Case("HIGHEST", Precision::HIGHEST)
      .Default(::std::nullopt);
}
::std::optional<Precision> symbolizePrecision(uint32_t value) {
  switch (value) {
  case 0: return Precision::DEFAULT;
  case 1: return Precision::HIGH;
  case 2: return Precision::HIGHEST;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyRngAlgorithm(RngAlgorithm val) {
  switch (val) {
    case RngAlgorithm::DEFAULT: return "DEFAULT";
    case RngAlgorithm::THREE_FRY: return "THREE_FRY";
    case RngAlgorithm::PHILOX: return "PHILOX";
  }
  return "";
}

::std::optional<RngAlgorithm> symbolizeRngAlgorithm(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RngAlgorithm>>(str)
      .Case("DEFAULT", RngAlgorithm::DEFAULT)
      .Case("THREE_FRY", RngAlgorithm::THREE_FRY)
      .Case("PHILOX", RngAlgorithm::PHILOX)
      .Default(::std::nullopt);
}
::std::optional<RngAlgorithm> symbolizeRngAlgorithm(uint32_t value) {
  switch (value) {
  case 0: return RngAlgorithm::DEFAULT;
  case 1: return RngAlgorithm::THREE_FRY;
  case 2: return RngAlgorithm::PHILOX;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyRngDistribution(RngDistribution val) {
  switch (val) {
    case RngDistribution::UNIFORM: return "UNIFORM";
    case RngDistribution::NORMAL: return "NORMAL";
  }
  return "";
}

::std::optional<RngDistribution> symbolizeRngDistribution(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RngDistribution>>(str)
      .Case("UNIFORM", RngDistribution::UNIFORM)
      .Case("NORMAL", RngDistribution::NORMAL)
      .Default(::std::nullopt);
}
::std::optional<RngDistribution> symbolizeRngDistribution(uint32_t value) {
  switch (value) {
  case 1: return RngDistribution::UNIFORM;
  case 2: return RngDistribution::NORMAL;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

namespace mlir {
namespace stablehlo {
::llvm::StringRef stringifyTranspose(Transpose val) {
  switch (val) {
    case Transpose::TRANSPOSE_INVALID: return "TRANSPOSE_INVALID";
    case Transpose::NO_TRANSPOSE: return "NO_TRANSPOSE";
    case Transpose::TRANSPOSE: return "TRANSPOSE";
    case Transpose::ADJOINT: return "ADJOINT";
  }
  return "";
}

::std::optional<Transpose> symbolizeTranspose(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Transpose>>(str)
      .Case("TRANSPOSE_INVALID", Transpose::TRANSPOSE_INVALID)
      .Case("NO_TRANSPOSE", Transpose::NO_TRANSPOSE)
      .Case("TRANSPOSE", Transpose::TRANSPOSE)
      .Case("ADJOINT", Transpose::ADJOINT)
      .Default(::std::nullopt);
}
::std::optional<Transpose> symbolizeTranspose(uint32_t value) {
  switch (value) {
  case 0: return Transpose::TRANSPOSE_INVALID;
  case 1: return Transpose::NO_TRANSPOSE;
  case 2: return Transpose::TRANSPOSE;
  case 3: return Transpose::ADJOINT;
  default: return ::std::nullopt;
  }
}

} // namespace stablehlo
} // namespace mlir

