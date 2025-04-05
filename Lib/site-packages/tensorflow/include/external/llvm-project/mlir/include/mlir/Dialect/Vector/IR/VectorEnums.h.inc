/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: VectorAttributes.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
// Kind of combining function for contractions and reductions
enum class CombiningKind : uint32_t {
  ADD = 0,
  MUL = 1,
  MINUI = 2,
  MINSI = 3,
  MINNUMF = 4,
  MAXUI = 5,
  MAXSI = 6,
  MAXNUMF = 7,
  AND = 8,
  OR = 9,
  XOR = 10,
  MAXIMUMF = 12,
  MINIMUMF = 11,
};

::std::optional<CombiningKind> symbolizeCombiningKind(uint32_t);
::llvm::StringRef stringifyCombiningKind(CombiningKind);
::std::optional<CombiningKind> symbolizeCombiningKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCombiningKind() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(CombiningKind enumValue) {
  return stringifyCombiningKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CombiningKind> symbolizeEnum<CombiningKind>(::llvm::StringRef str) {
  return symbolizeCombiningKind(str);
}
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::CombiningKind, ::mlir::vector::CombiningKind> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::CombiningKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Kind of combining function for contractions and reductions");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::CombiningKind> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::CombiningKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Kind of combining function for contractions and reductions specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::CombiningKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::CombiningKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::CombiningKind getEmptyKey() {
    return static_cast<::mlir::vector::CombiningKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::CombiningKind getTombstoneKey() {
    return static_cast<::mlir::vector::CombiningKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::CombiningKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::CombiningKind &lhs, const ::mlir::vector::CombiningKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vector {
// Punctuation for separating vectors or vector elements
enum class PrintPunctuation : uint32_t {
  NoPunctuation = 0,
  NewLine = 1,
  Comma = 2,
  Open = 3,
  Close = 4,
};

::std::optional<PrintPunctuation> symbolizePrintPunctuation(uint32_t);
::llvm::StringRef stringifyPrintPunctuation(PrintPunctuation);
::std::optional<PrintPunctuation> symbolizePrintPunctuation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPrintPunctuation() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(PrintPunctuation enumValue) {
  return stringifyPrintPunctuation(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PrintPunctuation> symbolizeEnum<PrintPunctuation>(::llvm::StringRef str) {
  return symbolizePrintPunctuation(str);
}
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::PrintPunctuation, ::mlir::vector::PrintPunctuation> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::PrintPunctuation> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Punctuation for separating vectors or vector elements");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::PrintPunctuation> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::PrintPunctuation>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Punctuation for separating vectors or vector elements specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::PrintPunctuation value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::PrintPunctuation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::PrintPunctuation getEmptyKey() {
    return static_cast<::mlir::vector::PrintPunctuation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::PrintPunctuation getTombstoneKey() {
    return static_cast<::mlir::vector::PrintPunctuation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::PrintPunctuation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::PrintPunctuation &lhs, const ::mlir::vector::PrintPunctuation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vector {
// Iterator type
enum class IteratorType : uint32_t {
  parallel = 0,
  reduction = 1,
};

::std::optional<IteratorType> symbolizeIteratorType(uint32_t);
::llvm::StringRef stringifyIteratorType(IteratorType);
::std::optional<IteratorType> symbolizeIteratorType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForIteratorType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(IteratorType enumValue) {
  return stringifyIteratorType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<IteratorType> symbolizeEnum<IteratorType>(::llvm::StringRef str) {
  return symbolizeIteratorType(str);
}
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::IteratorType, ::mlir::vector::IteratorType> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::IteratorType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Iterator type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::IteratorType> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::IteratorType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Iterator type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::IteratorType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::IteratorType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::IteratorType getEmptyKey() {
    return static_cast<::mlir::vector::IteratorType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::IteratorType getTombstoneKey() {
    return static_cast<::mlir::vector::IteratorType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::IteratorType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::IteratorType &lhs, const ::mlir::vector::IteratorType &rhs) {
    return lhs == rhs;
  }
};
}

