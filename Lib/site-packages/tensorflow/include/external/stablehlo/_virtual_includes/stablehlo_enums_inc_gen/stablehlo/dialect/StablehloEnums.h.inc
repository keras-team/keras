/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: StablehloOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace stablehlo {
// Which comparison operation to perform.
enum class ComparisonDirection : uint32_t {
  EQ = 0,
  NE = 1,
  GE = 2,
  GT = 3,
  LE = 4,
  LT = 5,
};

::std::optional<ComparisonDirection> symbolizeComparisonDirection(uint32_t);
::llvm::StringRef stringifyComparisonDirection(ComparisonDirection);
::std::optional<ComparisonDirection> symbolizeComparisonDirection(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparisonDirection() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(ComparisonDirection enumValue) {
  return stringifyComparisonDirection(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ComparisonDirection> symbolizeEnum<ComparisonDirection>(::llvm::StringRef str) {
  return symbolizeComparisonDirection(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::ComparisonDirection, ::mlir::stablehlo::ComparisonDirection> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::ComparisonDirection> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Which comparison operation to perform.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::ComparisonDirection> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::ComparisonDirection>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Which comparison operation to perform. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::ComparisonDirection value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::ComparisonDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::ComparisonDirection getEmptyKey() {
    return static_cast<::mlir::stablehlo::ComparisonDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::ComparisonDirection getTombstoneKey() {
    return static_cast<::mlir::stablehlo::ComparisonDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::ComparisonDirection &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::ComparisonDirection &lhs, const ::mlir::stablehlo::ComparisonDirection &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// Which comparison type to use.
enum class ComparisonType : uint32_t {
  NOTYPE = 0,
  FLOAT = 1,
  TOTALORDER = 2,
  SIGNED = 3,
  UNSIGNED = 4,
};

::std::optional<ComparisonType> symbolizeComparisonType(uint32_t);
::llvm::StringRef stringifyComparisonType(ComparisonType);
::std::optional<ComparisonType> symbolizeComparisonType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparisonType() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ComparisonType enumValue) {
  return stringifyComparisonType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ComparisonType> symbolizeEnum<ComparisonType>(::llvm::StringRef str) {
  return symbolizeComparisonType(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::ComparisonType, ::mlir::stablehlo::ComparisonType> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::ComparisonType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Which comparison type to use.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::ComparisonType> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::ComparisonType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Which comparison type to use. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::ComparisonType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::ComparisonType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::ComparisonType getEmptyKey() {
    return static_cast<::mlir::stablehlo::ComparisonType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::ComparisonType getTombstoneKey() {
    return static_cast<::mlir::stablehlo::ComparisonType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::ComparisonType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::ComparisonType &lhs, const ::mlir::stablehlo::ComparisonType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// Custom call API version
enum class CustomCallApiVersion : uint32_t {
  API_VERSION_UNSPECIFIED = 0,
  API_VERSION_ORIGINAL = 1,
  API_VERSION_STATUS_RETURNING = 2,
  API_VERSION_STATUS_RETURNING_UNIFIED = 3,
  API_VERSION_TYPED_FFI = 4,
};

::std::optional<CustomCallApiVersion> symbolizeCustomCallApiVersion(uint32_t);
::llvm::StringRef stringifyCustomCallApiVersion(CustomCallApiVersion);
::std::optional<CustomCallApiVersion> symbolizeCustomCallApiVersion(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCustomCallApiVersion() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(CustomCallApiVersion enumValue) {
  return stringifyCustomCallApiVersion(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CustomCallApiVersion> symbolizeEnum<CustomCallApiVersion>(::llvm::StringRef str) {
  return symbolizeCustomCallApiVersion(str);
}

class CustomCallApiVersionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CustomCallApiVersion;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CustomCallApiVersionAttr get(::mlir::MLIRContext *context, CustomCallApiVersion val);
  CustomCallApiVersion getValue() const;
};
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::CustomCallApiVersion, ::mlir::stablehlo::CustomCallApiVersion> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::CustomCallApiVersion> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Custom call API version");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::CustomCallApiVersion> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::CustomCallApiVersion>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Custom call API version specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::CustomCallApiVersion value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::CustomCallApiVersion> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::CustomCallApiVersion getEmptyKey() {
    return static_cast<::mlir::stablehlo::CustomCallApiVersion>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::CustomCallApiVersion getTombstoneKey() {
    return static_cast<::mlir::stablehlo::CustomCallApiVersion>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::CustomCallApiVersion &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::CustomCallApiVersion &lhs, const ::mlir::stablehlo::CustomCallApiVersion &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA fast fourier transform type.
enum class FftType : uint32_t {
  FFT = 0,
  IFFT = 1,
  RFFT = 2,
  IRFFT = 3,
};

::std::optional<FftType> symbolizeFftType(uint32_t);
::llvm::StringRef stringifyFftType(FftType);
::std::optional<FftType> symbolizeFftType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFftType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(FftType enumValue) {
  return stringifyFftType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FftType> symbolizeEnum<FftType>(::llvm::StringRef str) {
  return symbolizeFftType(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::FftType, ::mlir::stablehlo::FftType> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::FftType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XLA fast fourier transform type.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::FftType> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::FftType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid XLA fast fourier transform type. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::FftType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::FftType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::FftType getEmptyKey() {
    return static_cast<::mlir::stablehlo::FftType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::FftType getTombstoneKey() {
    return static_cast<::mlir::stablehlo::FftType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::FftType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::FftType &lhs, const ::mlir::stablehlo::FftType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA precision for an operand. Has backend specific meaning.
enum class Precision : uint32_t {
  DEFAULT = 0,
  HIGH = 1,
  HIGHEST = 2,
};

::std::optional<Precision> symbolizePrecision(uint32_t);
::llvm::StringRef stringifyPrecision(Precision);
::std::optional<Precision> symbolizePrecision(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPrecision() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Precision enumValue) {
  return stringifyPrecision(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Precision> symbolizeEnum<Precision>(::llvm::StringRef str) {
  return symbolizePrecision(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::Precision, ::mlir::stablehlo::Precision> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::Precision> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XLA precision for an operand. Has backend specific meaning.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::Precision> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::Precision>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid XLA precision for an operand. Has backend specific meaning. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::Precision value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::Precision> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::Precision getEmptyKey() {
    return static_cast<::mlir::stablehlo::Precision>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::Precision getTombstoneKey() {
    return static_cast<::mlir::stablehlo::Precision>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::Precision &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::Precision &lhs, const ::mlir::stablehlo::Precision &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA PRNG algorithm to be used.
enum class RngAlgorithm : uint32_t {
  DEFAULT = 0,
  THREE_FRY = 1,
  PHILOX = 2,
};

::std::optional<RngAlgorithm> symbolizeRngAlgorithm(uint32_t);
::llvm::StringRef stringifyRngAlgorithm(RngAlgorithm);
::std::optional<RngAlgorithm> symbolizeRngAlgorithm(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRngAlgorithm() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RngAlgorithm enumValue) {
  return stringifyRngAlgorithm(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RngAlgorithm> symbolizeEnum<RngAlgorithm>(::llvm::StringRef str) {
  return symbolizeRngAlgorithm(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::RngAlgorithm, ::mlir::stablehlo::RngAlgorithm> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::RngAlgorithm> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XLA PRNG algorithm to be used.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::RngAlgorithm> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::RngAlgorithm>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid XLA PRNG algorithm to be used. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::RngAlgorithm value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::RngAlgorithm> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::RngAlgorithm getEmptyKey() {
    return static_cast<::mlir::stablehlo::RngAlgorithm>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::RngAlgorithm getTombstoneKey() {
    return static_cast<::mlir::stablehlo::RngAlgorithm>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::RngAlgorithm &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::RngAlgorithm &lhs, const ::mlir::stablehlo::RngAlgorithm &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA PRNG distribution to be used.
enum class RngDistribution : uint32_t {
  UNIFORM = 1,
  NORMAL = 2,
};

::std::optional<RngDistribution> symbolizeRngDistribution(uint32_t);
::llvm::StringRef stringifyRngDistribution(RngDistribution);
::std::optional<RngDistribution> symbolizeRngDistribution(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRngDistribution() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RngDistribution enumValue) {
  return stringifyRngDistribution(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RngDistribution> symbolizeEnum<RngDistribution>(::llvm::StringRef str) {
  return symbolizeRngDistribution(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::RngDistribution, ::mlir::stablehlo::RngDistribution> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::RngDistribution> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XLA PRNG distribution to be used.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::RngDistribution> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::RngDistribution>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid XLA PRNG distribution to be used. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::RngDistribution value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::RngDistribution> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::RngDistribution getEmptyKey() {
    return static_cast<::mlir::stablehlo::RngDistribution>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::RngDistribution getTombstoneKey() {
    return static_cast<::mlir::stablehlo::RngDistribution>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::RngDistribution &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::RngDistribution &lhs, const ::mlir::stablehlo::RngDistribution &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// Transpose options
enum class Transpose : uint32_t {
  TRANSPOSE_INVALID = 0,
  NO_TRANSPOSE = 1,
  TRANSPOSE = 2,
  ADJOINT = 3,
};

::std::optional<Transpose> symbolizeTranspose(uint32_t);
::llvm::StringRef stringifyTranspose(Transpose);
::std::optional<Transpose> symbolizeTranspose(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTranspose() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(Transpose enumValue) {
  return stringifyTranspose(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Transpose> symbolizeEnum<Transpose>(::llvm::StringRef str) {
  return symbolizeTranspose(str);
}
} // namespace stablehlo
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::stablehlo::Transpose, ::mlir::stablehlo::Transpose> {
  template <typename ParserT>
  static FailureOr<::mlir::stablehlo::Transpose> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Transpose options");

    // Symbolize the keyword.
    if (::std::optional<::mlir::stablehlo::Transpose> attr = ::mlir::stablehlo::symbolizeEnum<::mlir::stablehlo::Transpose>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Transpose options specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::stablehlo::Transpose value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::Transpose> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::Transpose getEmptyKey() {
    return static_cast<::mlir::stablehlo::Transpose>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::Transpose getTombstoneKey() {
    return static_cast<::mlir::stablehlo::Transpose>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::Transpose &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::Transpose &lhs, const ::mlir::stablehlo::Transpose &rhs) {
    return lhs == rhs;
  }
};
}

