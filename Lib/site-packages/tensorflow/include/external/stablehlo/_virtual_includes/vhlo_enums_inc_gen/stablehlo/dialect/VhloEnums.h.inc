/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: VhloEnums.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vhlo {
// ComparisonDirectionV1
enum class ComparisonDirectionV1 : uint32_t {
  EQ = 0,
  NE = 1,
  GE = 2,
  GT = 3,
  LE = 4,
  LT = 5,
};

::std::optional<ComparisonDirectionV1> symbolizeComparisonDirectionV1(uint32_t);
::llvm::StringRef stringifyComparisonDirectionV1(ComparisonDirectionV1);
::std::optional<ComparisonDirectionV1> symbolizeComparisonDirectionV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparisonDirectionV1() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(ComparisonDirectionV1 enumValue) {
  return stringifyComparisonDirectionV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ComparisonDirectionV1> symbolizeEnum<ComparisonDirectionV1>(::llvm::StringRef str) {
  return symbolizeComparisonDirectionV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::ComparisonDirectionV1, ::mlir::vhlo::ComparisonDirectionV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::ComparisonDirectionV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ComparisonDirectionV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::ComparisonDirectionV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::ComparisonDirectionV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ComparisonDirectionV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::ComparisonDirectionV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::ComparisonDirectionV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::ComparisonDirectionV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::ComparisonDirectionV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::ComparisonDirectionV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::ComparisonDirectionV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::ComparisonDirectionV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::ComparisonDirectionV1 &lhs, const ::mlir::vhlo::ComparisonDirectionV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// ComparisonTypeV1
enum class ComparisonTypeV1 : uint32_t {
  NOTYPE = 0,
  FLOAT = 1,
  TOTALORDER = 2,
  SIGNED = 3,
  UNSIGNED = 4,
};

::std::optional<ComparisonTypeV1> symbolizeComparisonTypeV1(uint32_t);
::llvm::StringRef stringifyComparisonTypeV1(ComparisonTypeV1);
::std::optional<ComparisonTypeV1> symbolizeComparisonTypeV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparisonTypeV1() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ComparisonTypeV1 enumValue) {
  return stringifyComparisonTypeV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ComparisonTypeV1> symbolizeEnum<ComparisonTypeV1>(::llvm::StringRef str) {
  return symbolizeComparisonTypeV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::ComparisonTypeV1, ::mlir::vhlo::ComparisonTypeV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::ComparisonTypeV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ComparisonTypeV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::ComparisonTypeV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::ComparisonTypeV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ComparisonTypeV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::ComparisonTypeV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::ComparisonTypeV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::ComparisonTypeV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::ComparisonTypeV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::ComparisonTypeV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::ComparisonTypeV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::ComparisonTypeV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::ComparisonTypeV1 &lhs, const ::mlir::vhlo::ComparisonTypeV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// CustomCallApiVersionV1
enum class CustomCallApiVersionV1 : uint32_t {
  API_VERSION_UNSPECIFIED = 0,
  API_VERSION_ORIGINAL = 1,
  API_VERSION_STATUS_RETURNING = 2,
  API_VERSION_STATUS_RETURNING_UNIFIED = 3,
  API_VERSION_TYPED_FFI = 4,
};

::std::optional<CustomCallApiVersionV1> symbolizeCustomCallApiVersionV1(uint32_t);
::llvm::StringRef stringifyCustomCallApiVersionV1(CustomCallApiVersionV1);
::std::optional<CustomCallApiVersionV1> symbolizeCustomCallApiVersionV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCustomCallApiVersionV1() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(CustomCallApiVersionV1 enumValue) {
  return stringifyCustomCallApiVersionV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CustomCallApiVersionV1> symbolizeEnum<CustomCallApiVersionV1>(::llvm::StringRef str) {
  return symbolizeCustomCallApiVersionV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::CustomCallApiVersionV1, ::mlir::vhlo::CustomCallApiVersionV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::CustomCallApiVersionV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for CustomCallApiVersionV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::CustomCallApiVersionV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::CustomCallApiVersionV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid CustomCallApiVersionV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::CustomCallApiVersionV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::CustomCallApiVersionV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::CustomCallApiVersionV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::CustomCallApiVersionV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::CustomCallApiVersionV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::CustomCallApiVersionV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::CustomCallApiVersionV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::CustomCallApiVersionV1 &lhs, const ::mlir::vhlo::CustomCallApiVersionV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// FftTypeV1
enum class FftTypeV1 : uint32_t {
  FFT = 0,
  IFFT = 1,
  RFFT = 2,
  IRFFT = 3,
};

::std::optional<FftTypeV1> symbolizeFftTypeV1(uint32_t);
::llvm::StringRef stringifyFftTypeV1(FftTypeV1);
::std::optional<FftTypeV1> symbolizeFftTypeV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFftTypeV1() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(FftTypeV1 enumValue) {
  return stringifyFftTypeV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FftTypeV1> symbolizeEnum<FftTypeV1>(::llvm::StringRef str) {
  return symbolizeFftTypeV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::FftTypeV1, ::mlir::vhlo::FftTypeV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::FftTypeV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for FftTypeV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::FftTypeV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::FftTypeV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid FftTypeV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::FftTypeV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::FftTypeV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::FftTypeV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::FftTypeV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::FftTypeV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::FftTypeV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::FftTypeV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::FftTypeV1 &lhs, const ::mlir::vhlo::FftTypeV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// PrecisionV1
enum class PrecisionV1 : uint32_t {
  DEFAULT = 0,
  HIGH = 1,
  HIGHEST = 2,
};

::std::optional<PrecisionV1> symbolizePrecisionV1(uint32_t);
::llvm::StringRef stringifyPrecisionV1(PrecisionV1);
::std::optional<PrecisionV1> symbolizePrecisionV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPrecisionV1() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(PrecisionV1 enumValue) {
  return stringifyPrecisionV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PrecisionV1> symbolizeEnum<PrecisionV1>(::llvm::StringRef str) {
  return symbolizePrecisionV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::PrecisionV1, ::mlir::vhlo::PrecisionV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::PrecisionV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PrecisionV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::PrecisionV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::PrecisionV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PrecisionV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::PrecisionV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::PrecisionV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::PrecisionV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::PrecisionV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::PrecisionV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::PrecisionV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::PrecisionV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::PrecisionV1 &lhs, const ::mlir::vhlo::PrecisionV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// RngAlgorithmV1
enum class RngAlgorithmV1 : uint32_t {
  DEFAULT = 0,
  THREE_FRY = 1,
  PHILOX = 2,
};

::std::optional<RngAlgorithmV1> symbolizeRngAlgorithmV1(uint32_t);
::llvm::StringRef stringifyRngAlgorithmV1(RngAlgorithmV1);
::std::optional<RngAlgorithmV1> symbolizeRngAlgorithmV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRngAlgorithmV1() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RngAlgorithmV1 enumValue) {
  return stringifyRngAlgorithmV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RngAlgorithmV1> symbolizeEnum<RngAlgorithmV1>(::llvm::StringRef str) {
  return symbolizeRngAlgorithmV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::RngAlgorithmV1, ::mlir::vhlo::RngAlgorithmV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::RngAlgorithmV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for RngAlgorithmV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::RngAlgorithmV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::RngAlgorithmV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid RngAlgorithmV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::RngAlgorithmV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::RngAlgorithmV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::RngAlgorithmV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::RngAlgorithmV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::RngAlgorithmV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::RngAlgorithmV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::RngAlgorithmV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::RngAlgorithmV1 &lhs, const ::mlir::vhlo::RngAlgorithmV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// RngDistributionV1
enum class RngDistributionV1 : uint32_t {
  UNIFORM = 1,
  NORMAL = 2,
};

::std::optional<RngDistributionV1> symbolizeRngDistributionV1(uint32_t);
::llvm::StringRef stringifyRngDistributionV1(RngDistributionV1);
::std::optional<RngDistributionV1> symbolizeRngDistributionV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRngDistributionV1() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RngDistributionV1 enumValue) {
  return stringifyRngDistributionV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RngDistributionV1> symbolizeEnum<RngDistributionV1>(::llvm::StringRef str) {
  return symbolizeRngDistributionV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::RngDistributionV1, ::mlir::vhlo::RngDistributionV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::RngDistributionV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for RngDistributionV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::RngDistributionV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::RngDistributionV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid RngDistributionV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::RngDistributionV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::RngDistributionV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::RngDistributionV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::RngDistributionV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::RngDistributionV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::RngDistributionV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::RngDistributionV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::RngDistributionV1 &lhs, const ::mlir::vhlo::RngDistributionV1 &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vhlo {
// TransposeV1
enum class TransposeV1 : uint32_t {
  TRANSPOSE_INVALID = 0,
  NO_TRANSPOSE = 1,
  TRANSPOSE = 2,
  ADJOINT = 3,
};

::std::optional<TransposeV1> symbolizeTransposeV1(uint32_t);
::llvm::StringRef stringifyTransposeV1(TransposeV1);
::std::optional<TransposeV1> symbolizeTransposeV1(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTransposeV1() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(TransposeV1 enumValue) {
  return stringifyTransposeV1(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TransposeV1> symbolizeEnum<TransposeV1>(::llvm::StringRef str) {
  return symbolizeTransposeV1(str);
}
} // namespace vhlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vhlo::TransposeV1, ::mlir::vhlo::TransposeV1> {
  template <typename ParserT>
  static FailureOr<::mlir::vhlo::TransposeV1> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for TransposeV1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vhlo::TransposeV1> attr = ::mlir::vhlo::symbolizeEnum<::mlir::vhlo::TransposeV1>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid TransposeV1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vhlo::TransposeV1 value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vhlo::TransposeV1> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vhlo::TransposeV1 getEmptyKey() {
    return static_cast<::mlir::vhlo::TransposeV1>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vhlo::TransposeV1 getTombstoneKey() {
    return static_cast<::mlir::vhlo::TransposeV1>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vhlo::TransposeV1 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vhlo::TransposeV1 &lhs, const ::mlir::vhlo::TransposeV1 &rhs) {
    return lhs == rhs;
  }
};
}

