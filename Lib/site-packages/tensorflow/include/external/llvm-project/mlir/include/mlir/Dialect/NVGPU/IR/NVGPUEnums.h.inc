/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPU.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace nvgpu {
// Rounding mode of rcp
enum class RcpRoundingMode : uint32_t {
  APPROX = 0,
  RN = 1,
  RZ = 2,
  RM = 3,
  RP = 4,
};

::std::optional<RcpRoundingMode> symbolizeRcpRoundingMode(uint32_t);
::llvm::StringRef stringifyRcpRoundingMode(RcpRoundingMode);
::std::optional<RcpRoundingMode> symbolizeRcpRoundingMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRcpRoundingMode() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(RcpRoundingMode enumValue) {
  return stringifyRcpRoundingMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RcpRoundingMode> symbolizeEnum<RcpRoundingMode>(::llvm::StringRef str) {
  return symbolizeRcpRoundingMode(str);
}
} // namespace nvgpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::nvgpu::RcpRoundingMode, ::mlir::nvgpu::RcpRoundingMode> {
  template <typename ParserT>
  static FailureOr<::mlir::nvgpu::RcpRoundingMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Rounding mode of rcp");

    // Symbolize the keyword.
    if (::std::optional<::mlir::nvgpu::RcpRoundingMode> attr = ::mlir::nvgpu::symbolizeEnum<::mlir::nvgpu::RcpRoundingMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Rounding mode of rcp specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::nvgpu::RcpRoundingMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::nvgpu::RcpRoundingMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::nvgpu::RcpRoundingMode getEmptyKey() {
    return static_cast<::mlir::nvgpu::RcpRoundingMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::nvgpu::RcpRoundingMode getTombstoneKey() {
    return static_cast<::mlir::nvgpu::RcpRoundingMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::nvgpu::RcpRoundingMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::nvgpu::RcpRoundingMode &lhs, const ::mlir::nvgpu::RcpRoundingMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace nvgpu {
// Tensor map interleave layout type
enum class TensorMapInterleaveKind : uint32_t {
  INTERLEAVE_NONE = 0,
  INTERLEAVE_16B = 1,
  INTERLEAVE_32B = 2,
};

::std::optional<TensorMapInterleaveKind> symbolizeTensorMapInterleaveKind(uint32_t);
::llvm::StringRef stringifyTensorMapInterleaveKind(TensorMapInterleaveKind);
::std::optional<TensorMapInterleaveKind> symbolizeTensorMapInterleaveKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTensorMapInterleaveKind() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(TensorMapInterleaveKind enumValue) {
  return stringifyTensorMapInterleaveKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TensorMapInterleaveKind> symbolizeEnum<TensorMapInterleaveKind>(::llvm::StringRef str) {
  return symbolizeTensorMapInterleaveKind(str);
}
} // namespace nvgpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::nvgpu::TensorMapInterleaveKind, ::mlir::nvgpu::TensorMapInterleaveKind> {
  template <typename ParserT>
  static FailureOr<::mlir::nvgpu::TensorMapInterleaveKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Tensor map interleave layout type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::nvgpu::TensorMapInterleaveKind> attr = ::mlir::nvgpu::symbolizeEnum<::mlir::nvgpu::TensorMapInterleaveKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Tensor map interleave layout type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::nvgpu::TensorMapInterleaveKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::nvgpu::TensorMapInterleaveKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::nvgpu::TensorMapInterleaveKind getEmptyKey() {
    return static_cast<::mlir::nvgpu::TensorMapInterleaveKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::nvgpu::TensorMapInterleaveKind getTombstoneKey() {
    return static_cast<::mlir::nvgpu::TensorMapInterleaveKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::nvgpu::TensorMapInterleaveKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::nvgpu::TensorMapInterleaveKind &lhs, const ::mlir::nvgpu::TensorMapInterleaveKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace nvgpu {
// Tensor map L2 promotion type
enum class TensorMapL2PromoKind : uint32_t {
  L2PROMO_NONE = 0,
  L2PROMO_64B = 1,
  L2PROMO_128B = 2,
  L2PROMO_256B = 3,
};

::std::optional<TensorMapL2PromoKind> symbolizeTensorMapL2PromoKind(uint32_t);
::llvm::StringRef stringifyTensorMapL2PromoKind(TensorMapL2PromoKind);
::std::optional<TensorMapL2PromoKind> symbolizeTensorMapL2PromoKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTensorMapL2PromoKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(TensorMapL2PromoKind enumValue) {
  return stringifyTensorMapL2PromoKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TensorMapL2PromoKind> symbolizeEnum<TensorMapL2PromoKind>(::llvm::StringRef str) {
  return symbolizeTensorMapL2PromoKind(str);
}
} // namespace nvgpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::nvgpu::TensorMapL2PromoKind, ::mlir::nvgpu::TensorMapL2PromoKind> {
  template <typename ParserT>
  static FailureOr<::mlir::nvgpu::TensorMapL2PromoKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Tensor map L2 promotion type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::nvgpu::TensorMapL2PromoKind> attr = ::mlir::nvgpu::symbolizeEnum<::mlir::nvgpu::TensorMapL2PromoKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Tensor map L2 promotion type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::nvgpu::TensorMapL2PromoKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::nvgpu::TensorMapL2PromoKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::nvgpu::TensorMapL2PromoKind getEmptyKey() {
    return static_cast<::mlir::nvgpu::TensorMapL2PromoKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::nvgpu::TensorMapL2PromoKind getTombstoneKey() {
    return static_cast<::mlir::nvgpu::TensorMapL2PromoKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::nvgpu::TensorMapL2PromoKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::nvgpu::TensorMapL2PromoKind &lhs, const ::mlir::nvgpu::TensorMapL2PromoKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace nvgpu {
// Tensor map out-of-bounds fill type
enum class TensorMapOOBKind : uint32_t {
  OOB_ZERO = 0,
  OOB_NAN = 1,
};

::std::optional<TensorMapOOBKind> symbolizeTensorMapOOBKind(uint32_t);
::llvm::StringRef stringifyTensorMapOOBKind(TensorMapOOBKind);
::std::optional<TensorMapOOBKind> symbolizeTensorMapOOBKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTensorMapOOBKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(TensorMapOOBKind enumValue) {
  return stringifyTensorMapOOBKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TensorMapOOBKind> symbolizeEnum<TensorMapOOBKind>(::llvm::StringRef str) {
  return symbolizeTensorMapOOBKind(str);
}
} // namespace nvgpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::nvgpu::TensorMapOOBKind, ::mlir::nvgpu::TensorMapOOBKind> {
  template <typename ParserT>
  static FailureOr<::mlir::nvgpu::TensorMapOOBKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Tensor map out-of-bounds fill type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::nvgpu::TensorMapOOBKind> attr = ::mlir::nvgpu::symbolizeEnum<::mlir::nvgpu::TensorMapOOBKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Tensor map out-of-bounds fill type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::nvgpu::TensorMapOOBKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::nvgpu::TensorMapOOBKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::nvgpu::TensorMapOOBKind getEmptyKey() {
    return static_cast<::mlir::nvgpu::TensorMapOOBKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::nvgpu::TensorMapOOBKind getTombstoneKey() {
    return static_cast<::mlir::nvgpu::TensorMapOOBKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::nvgpu::TensorMapOOBKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::nvgpu::TensorMapOOBKind &lhs, const ::mlir::nvgpu::TensorMapOOBKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace nvgpu {
// Tensor map swizzling mode of shared memory banks
enum class TensorMapSwizzleKind : uint32_t {
  SWIZZLE_NONE = 0,
  SWIZZLE_32B = 1,
  SWIZZLE_64B = 2,
  SWIZZLE_128B = 3,
};

::std::optional<TensorMapSwizzleKind> symbolizeTensorMapSwizzleKind(uint32_t);
::llvm::StringRef stringifyTensorMapSwizzleKind(TensorMapSwizzleKind);
::std::optional<TensorMapSwizzleKind> symbolizeTensorMapSwizzleKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTensorMapSwizzleKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(TensorMapSwizzleKind enumValue) {
  return stringifyTensorMapSwizzleKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TensorMapSwizzleKind> symbolizeEnum<TensorMapSwizzleKind>(::llvm::StringRef str) {
  return symbolizeTensorMapSwizzleKind(str);
}
} // namespace nvgpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::nvgpu::TensorMapSwizzleKind, ::mlir::nvgpu::TensorMapSwizzleKind> {
  template <typename ParserT>
  static FailureOr<::mlir::nvgpu::TensorMapSwizzleKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Tensor map swizzling mode of shared memory banks");

    // Symbolize the keyword.
    if (::std::optional<::mlir::nvgpu::TensorMapSwizzleKind> attr = ::mlir::nvgpu::symbolizeEnum<::mlir::nvgpu::TensorMapSwizzleKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Tensor map swizzling mode of shared memory banks specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::nvgpu::TensorMapSwizzleKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::nvgpu::TensorMapSwizzleKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::nvgpu::TensorMapSwizzleKind getEmptyKey() {
    return static_cast<::mlir::nvgpu::TensorMapSwizzleKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::nvgpu::TensorMapSwizzleKind getTombstoneKey() {
    return static_cast<::mlir::nvgpu::TensorMapSwizzleKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::nvgpu::TensorMapSwizzleKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::nvgpu::TensorMapSwizzleKind &lhs, const ::mlir::nvgpu::TensorMapSwizzleKind &rhs) {
    return lhs == rhs;
  }
};
}

