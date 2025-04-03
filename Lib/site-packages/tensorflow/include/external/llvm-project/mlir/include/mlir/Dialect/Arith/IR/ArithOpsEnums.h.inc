/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArithBase.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace arith {
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
enum class CmpFPredicate : uint64_t {
  AlwaysFalse = 0,
  OEQ = 1,
  OGT = 2,
  OGE = 3,
  OLT = 4,
  OLE = 5,
  ONE = 6,
  ORD = 7,
  UEQ = 8,
  UGT = 9,
  UGE = 10,
  ULT = 11,
  ULE = 12,
  UNE = 13,
  UNO = 14,
  AlwaysTrue = 15,
};

::std::optional<CmpFPredicate> symbolizeCmpFPredicate(uint64_t);
::llvm::StringRef stringifyCmpFPredicate(CmpFPredicate);
::std::optional<CmpFPredicate> symbolizeCmpFPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCmpFPredicate() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(CmpFPredicate enumValue) {
  return stringifyCmpFPredicate(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CmpFPredicate> symbolizeEnum<CmpFPredicate>(::llvm::StringRef str) {
  return symbolizeCmpFPredicate(str);
}

class CmpFPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CmpFPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CmpFPredicateAttr get(::mlir::MLIRContext *context, CmpFPredicate val);
  CmpFPredicate getValue() const;
};
} // namespace arith
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::arith::CmpFPredicate, ::mlir::arith::CmpFPredicate> {
  template <typename ParserT>
  static FailureOr<::mlir::arith::CmpFPredicate> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15");

    // Symbolize the keyword.
    if (::std::optional<::mlir::arith::CmpFPredicate> attr = ::mlir::arith::symbolizeEnum<::mlir::arith::CmpFPredicate>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::arith::CmpFPredicate value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::CmpFPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::arith::CmpFPredicate getEmptyKey() {
    return static_cast<::mlir::arith::CmpFPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::CmpFPredicate getTombstoneKey() {
    return static_cast<::mlir::arith::CmpFPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::CmpFPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::arith::CmpFPredicate &lhs, const ::mlir::arith::CmpFPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
enum class CmpIPredicate : uint64_t {
  eq = 0,
  ne = 1,
  slt = 2,
  sle = 3,
  sgt = 4,
  sge = 5,
  ult = 6,
  ule = 7,
  ugt = 8,
  uge = 9,
};

::std::optional<CmpIPredicate> symbolizeCmpIPredicate(uint64_t);
::llvm::StringRef stringifyCmpIPredicate(CmpIPredicate);
::std::optional<CmpIPredicate> symbolizeCmpIPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCmpIPredicate() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(CmpIPredicate enumValue) {
  return stringifyCmpIPredicate(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CmpIPredicate> symbolizeEnum<CmpIPredicate>(::llvm::StringRef str) {
  return symbolizeCmpIPredicate(str);
}

class CmpIPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CmpIPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CmpIPredicateAttr get(::mlir::MLIRContext *context, CmpIPredicate val);
  CmpIPredicate getValue() const;
};
} // namespace arith
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::arith::CmpIPredicate, ::mlir::arith::CmpIPredicate> {
  template <typename ParserT>
  static FailureOr<::mlir::arith::CmpIPredicate> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");

    // Symbolize the keyword.
    if (::std::optional<::mlir::arith::CmpIPredicate> attr = ::mlir::arith::symbolizeEnum<::mlir::arith::CmpIPredicate>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::arith::CmpIPredicate value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::CmpIPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::arith::CmpIPredicate getEmptyKey() {
    return static_cast<::mlir::arith::CmpIPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::CmpIPredicate getTombstoneKey() {
    return static_cast<::mlir::arith::CmpIPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::CmpIPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::arith::CmpIPredicate &lhs, const ::mlir::arith::CmpIPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// Integer overflow arith flags
enum class IntegerOverflowFlags : uint32_t {
  none = 0,
  nsw = 1,
  nuw = 2,
};

::std::optional<IntegerOverflowFlags> symbolizeIntegerOverflowFlags(uint32_t);
std::string stringifyIntegerOverflowFlags(IntegerOverflowFlags);
::std::optional<IntegerOverflowFlags> symbolizeIntegerOverflowFlags(::llvm::StringRef);

inline constexpr IntegerOverflowFlags operator|(IntegerOverflowFlags a, IntegerOverflowFlags b) {
  return static_cast<IntegerOverflowFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr IntegerOverflowFlags operator&(IntegerOverflowFlags a, IntegerOverflowFlags b) {
  return static_cast<IntegerOverflowFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr IntegerOverflowFlags operator^(IntegerOverflowFlags a, IntegerOverflowFlags b) {
  return static_cast<IntegerOverflowFlags>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr IntegerOverflowFlags operator~(IntegerOverflowFlags bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<IntegerOverflowFlags>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(3u));
}
inline constexpr bool bitEnumContainsAll(IntegerOverflowFlags bits, IntegerOverflowFlags bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(IntegerOverflowFlags bits, IntegerOverflowFlags bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr IntegerOverflowFlags bitEnumClear(IntegerOverflowFlags bits, IntegerOverflowFlags bit) {
  return bits & ~bit;
}
inline constexpr IntegerOverflowFlags bitEnumSet(IntegerOverflowFlags bits, IntegerOverflowFlags bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(IntegerOverflowFlags enumValue) {
  return stringifyIntegerOverflowFlags(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<IntegerOverflowFlags> symbolizeEnum<IntegerOverflowFlags>(::llvm::StringRef str) {
  return symbolizeIntegerOverflowFlags(str);
}
} // namespace arith
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::arith::IntegerOverflowFlags, ::mlir::arith::IntegerOverflowFlags> {
  template <typename ParserT>
  static FailureOr<::mlir::arith::IntegerOverflowFlags> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Integer overflow arith flags");

    // Symbolize the keyword.
    if (::std::optional<::mlir::arith::IntegerOverflowFlags> attr = ::mlir::arith::symbolizeEnum<::mlir::arith::IntegerOverflowFlags>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Integer overflow arith flags specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::arith::IntegerOverflowFlags value) {
  auto valueStr = stringifyEnum(value);
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::arith::IntegerOverflowFlags>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::IntegerOverflowFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::arith::IntegerOverflowFlags getEmptyKey() {
    return static_cast<::mlir::arith::IntegerOverflowFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::IntegerOverflowFlags getTombstoneKey() {
    return static_cast<::mlir::arith::IntegerOverflowFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::IntegerOverflowFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::arith::IntegerOverflowFlags &lhs, const ::mlir::arith::IntegerOverflowFlags &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// Floating point rounding mode
enum class RoundingMode : uint32_t {
  to_nearest_even = 0,
  downward = 1,
  upward = 2,
  toward_zero = 3,
  to_nearest_away = 4,
};

::std::optional<RoundingMode> symbolizeRoundingMode(uint32_t);
::llvm::StringRef stringifyRoundingMode(RoundingMode);
::std::optional<RoundingMode> symbolizeRoundingMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRoundingMode() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(RoundingMode enumValue) {
  return stringifyRoundingMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RoundingMode> symbolizeEnum<RoundingMode>(::llvm::StringRef str) {
  return symbolizeRoundingMode(str);
}

class RoundingModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = RoundingMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static RoundingModeAttr get(::mlir::MLIRContext *context, RoundingMode val);
  RoundingMode getValue() const;
};
} // namespace arith
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::arith::RoundingMode, ::mlir::arith::RoundingMode> {
  template <typename ParserT>
  static FailureOr<::mlir::arith::RoundingMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Floating point rounding mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::arith::RoundingMode> attr = ::mlir::arith::symbolizeEnum<::mlir::arith::RoundingMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Floating point rounding mode specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::arith::RoundingMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::RoundingMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::arith::RoundingMode getEmptyKey() {
    return static_cast<::mlir::arith::RoundingMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::RoundingMode getTombstoneKey() {
    return static_cast<::mlir::arith::RoundingMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::RoundingMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::arith::RoundingMode &lhs, const ::mlir::arith::RoundingMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
enum class AtomicRMWKind : uint64_t {
  addf = 0,
  addi = 1,
  assign = 2,
  maximumf = 3,
  maxs = 4,
  maxu = 5,
  minimumf = 6,
  mins = 7,
  minu = 8,
  mulf = 9,
  muli = 10,
  ori = 11,
  andi = 12,
  maxnumf = 13,
  minnumf = 14,
};

::std::optional<AtomicRMWKind> symbolizeAtomicRMWKind(uint64_t);
::llvm::StringRef stringifyAtomicRMWKind(AtomicRMWKind);
::std::optional<AtomicRMWKind> symbolizeAtomicRMWKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicRMWKind() {
  return 14;
}


inline ::llvm::StringRef stringifyEnum(AtomicRMWKind enumValue) {
  return stringifyAtomicRMWKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AtomicRMWKind> symbolizeEnum<AtomicRMWKind>(::llvm::StringRef str) {
  return symbolizeAtomicRMWKind(str);
}

class AtomicRMWKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AtomicRMWKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AtomicRMWKindAttr get(::mlir::MLIRContext *context, AtomicRMWKind val);
  AtomicRMWKind getValue() const;
};
} // namespace arith
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::arith::AtomicRMWKind, ::mlir::arith::AtomicRMWKind> {
  template <typename ParserT>
  static FailureOr<::mlir::arith::AtomicRMWKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14");

    // Symbolize the keyword.
    if (::std::optional<::mlir::arith::AtomicRMWKind> attr = ::mlir::arith::symbolizeEnum<::mlir::arith::AtomicRMWKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::arith::AtomicRMWKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::AtomicRMWKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::arith::AtomicRMWKind getEmptyKey() {
    return static_cast<::mlir::arith::AtomicRMWKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::AtomicRMWKind getTombstoneKey() {
    return static_cast<::mlir::arith::AtomicRMWKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::AtomicRMWKind &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::arith::AtomicRMWKind &lhs, const ::mlir::arith::AtomicRMWKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace arith {
// Floating point fast math flags
enum class FastMathFlags : uint32_t {
  none = 0,
  reassoc = 1,
  nnan = 2,
  ninf = 4,
  nsz = 8,
  arcp = 16,
  contract = 32,
  afn = 64,
  fast = 127,
};

::std::optional<FastMathFlags> symbolizeFastMathFlags(uint32_t);
std::string stringifyFastMathFlags(FastMathFlags);
::std::optional<FastMathFlags> symbolizeFastMathFlags(::llvm::StringRef);

inline constexpr FastMathFlags operator|(FastMathFlags a, FastMathFlags b) {
  return static_cast<FastMathFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr FastMathFlags operator&(FastMathFlags a, FastMathFlags b) {
  return static_cast<FastMathFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr FastMathFlags operator^(FastMathFlags a, FastMathFlags b) {
  return static_cast<FastMathFlags>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr FastMathFlags operator~(FastMathFlags bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<FastMathFlags>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(127u));
}
inline constexpr bool bitEnumContainsAll(FastMathFlags bits, FastMathFlags bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(FastMathFlags bits, FastMathFlags bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr FastMathFlags bitEnumClear(FastMathFlags bits, FastMathFlags bit) {
  return bits & ~bit;
}
inline constexpr FastMathFlags bitEnumSet(FastMathFlags bits, FastMathFlags bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(FastMathFlags enumValue) {
  return stringifyFastMathFlags(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FastMathFlags> symbolizeEnum<FastMathFlags>(::llvm::StringRef str) {
  return symbolizeFastMathFlags(str);
}
} // namespace arith
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::arith::FastMathFlags, ::mlir::arith::FastMathFlags> {
  template <typename ParserT>
  static FailureOr<::mlir::arith::FastMathFlags> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Floating point fast math flags");

    // Symbolize the keyword.
    if (::std::optional<::mlir::arith::FastMathFlags> attr = ::mlir::arith::symbolizeEnum<::mlir::arith::FastMathFlags>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Floating point fast math flags specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::arith::FastMathFlags value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::arith::FastMathFlags::fast:
    return p << valueStr;
  default:
    break;
  }
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::arith::FastMathFlags>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::arith::FastMathFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::arith::FastMathFlags getEmptyKey() {
    return static_cast<::mlir::arith::FastMathFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::arith::FastMathFlags getTombstoneKey() {
    return static_cast<::mlir::arith::FastMathFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::arith::FastMathFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::arith::FastMathFlags &lhs, const ::mlir::arith::FastMathFlags &rhs) {
    return lhs == rhs;
  }
};
}

