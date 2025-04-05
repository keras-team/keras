/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: LinalgEnums.td                                                       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {
// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
enum class BinaryFn : uint32_t {
  add = 0,
  sub = 1,
  mul = 2,
  div = 3,
  div_unsigned = 4,
  max_signed = 5,
  min_signed = 6,
  max_unsigned = 7,
  min_unsigned = 8,
  powf = 9,
};

::std::optional<BinaryFn> symbolizeBinaryFn(uint32_t);
::llvm::StringRef stringifyBinaryFn(BinaryFn);
::std::optional<BinaryFn> symbolizeBinaryFn(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBinaryFn() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(BinaryFn enumValue) {
  return stringifyBinaryFn(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<BinaryFn> symbolizeEnum<BinaryFn>(::llvm::StringRef str) {
  return symbolizeBinaryFn(str);
}
} // namespace linalg
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::linalg::BinaryFn, ::mlir::linalg::BinaryFn> {
  template <typename ParserT>
  static FailureOr<::mlir::linalg::BinaryFn> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");

    // Symbolize the keyword.
    if (::std::optional<::mlir::linalg::BinaryFn> attr = ::mlir::linalg::symbolizeEnum<::mlir::linalg::BinaryFn>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::linalg::BinaryFn value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::linalg::BinaryFn> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::linalg::BinaryFn getEmptyKey() {
    return static_cast<::mlir::linalg::BinaryFn>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::linalg::BinaryFn getTombstoneKey() {
    return static_cast<::mlir::linalg::BinaryFn>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::linalg::BinaryFn &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::linalg::BinaryFn &lhs, const ::mlir::linalg::BinaryFn &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace linalg {
// allowed 32-bit signless integer cases: 0
enum class TernaryFn : uint32_t {
  select = 0,
};

::std::optional<TernaryFn> symbolizeTernaryFn(uint32_t);
::llvm::StringRef stringifyTernaryFn(TernaryFn);
::std::optional<TernaryFn> symbolizeTernaryFn(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTernaryFn() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(TernaryFn enumValue) {
  return stringifyTernaryFn(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TernaryFn> symbolizeEnum<TernaryFn>(::llvm::StringRef str) {
  return symbolizeTernaryFn(str);
}
} // namespace linalg
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::linalg::TernaryFn, ::mlir::linalg::TernaryFn> {
  template <typename ParserT>
  static FailureOr<::mlir::linalg::TernaryFn> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0");

    // Symbolize the keyword.
    if (::std::optional<::mlir::linalg::TernaryFn> attr = ::mlir::linalg::symbolizeEnum<::mlir::linalg::TernaryFn>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 32-bit signless integer cases: 0 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::linalg::TernaryFn value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::linalg::TernaryFn> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::linalg::TernaryFn getEmptyKey() {
    return static_cast<::mlir::linalg::TernaryFn>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::linalg::TernaryFn getTombstoneKey() {
    return static_cast<::mlir::linalg::TernaryFn>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::linalg::TernaryFn &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::linalg::TernaryFn &lhs, const ::mlir::linalg::TernaryFn &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace linalg {
// allowed 32-bit signless integer cases: 0, 1
enum class TypeFn : uint32_t {
  cast_signed = 0,
  cast_unsigned = 1,
};

::std::optional<TypeFn> symbolizeTypeFn(uint32_t);
::llvm::StringRef stringifyTypeFn(TypeFn);
::std::optional<TypeFn> symbolizeTypeFn(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTypeFn() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(TypeFn enumValue) {
  return stringifyTypeFn(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TypeFn> symbolizeEnum<TypeFn>(::llvm::StringRef str) {
  return symbolizeTypeFn(str);
}
} // namespace linalg
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::linalg::TypeFn, ::mlir::linalg::TypeFn> {
  template <typename ParserT>
  static FailureOr<::mlir::linalg::TypeFn> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::linalg::TypeFn> attr = ::mlir::linalg::symbolizeEnum<::mlir::linalg::TypeFn>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 32-bit signless integer cases: 0, 1 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::linalg::TypeFn value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::linalg::TypeFn> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::linalg::TypeFn getEmptyKey() {
    return static_cast<::mlir::linalg::TypeFn>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::linalg::TypeFn getTombstoneKey() {
    return static_cast<::mlir::linalg::TypeFn>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::linalg::TypeFn &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::linalg::TypeFn &lhs, const ::mlir::linalg::TypeFn &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace linalg {
// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
enum class UnaryFn : uint32_t {
  exp = 0,
  log = 1,
  abs = 2,
  ceil = 3,
  floor = 4,
  negf = 5,
  reciprocal = 6,
  round = 7,
  sqrt = 8,
  rsqrt = 9,
  square = 10,
  tanh = 11,
  erf = 12,
};

::std::optional<UnaryFn> symbolizeUnaryFn(uint32_t);
::llvm::StringRef stringifyUnaryFn(UnaryFn);
::std::optional<UnaryFn> symbolizeUnaryFn(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForUnaryFn() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(UnaryFn enumValue) {
  return stringifyUnaryFn(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<UnaryFn> symbolizeEnum<UnaryFn>(::llvm::StringRef str) {
  return symbolizeUnaryFn(str);
}
} // namespace linalg
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::linalg::UnaryFn, ::mlir::linalg::UnaryFn> {
  template <typename ParserT>
  static FailureOr<::mlir::linalg::UnaryFn> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12");

    // Symbolize the keyword.
    if (::std::optional<::mlir::linalg::UnaryFn> attr = ::mlir::linalg::symbolizeEnum<::mlir::linalg::UnaryFn>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::linalg::UnaryFn value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::linalg::UnaryFn> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::linalg::UnaryFn getEmptyKey() {
    return static_cast<::mlir::linalg::UnaryFn>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::linalg::UnaryFn getTombstoneKey() {
    return static_cast<::mlir::linalg::UnaryFn>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::linalg::UnaryFn &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::linalg::UnaryFn &lhs, const ::mlir::linalg::UnaryFn &rhs) {
    return lhs == rhs;
  }
};
}

