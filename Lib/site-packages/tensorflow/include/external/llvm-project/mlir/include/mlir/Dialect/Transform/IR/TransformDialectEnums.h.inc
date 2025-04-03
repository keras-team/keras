/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TransformAttrs.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
// Silenceable error propagation policy
enum class FailurePropagationMode : uint32_t {
  Propagate = 1,
  Suppress = 2,
};

::std::optional<FailurePropagationMode> symbolizeFailurePropagationMode(uint32_t);
::llvm::StringRef stringifyFailurePropagationMode(FailurePropagationMode);
::std::optional<FailurePropagationMode> symbolizeFailurePropagationMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFailurePropagationMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(FailurePropagationMode enumValue) {
  return stringifyFailurePropagationMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FailurePropagationMode> symbolizeEnum<FailurePropagationMode>(::llvm::StringRef str) {
  return symbolizeFailurePropagationMode(str);
}

class FailurePropagationModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FailurePropagationMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FailurePropagationModeAttr get(::mlir::MLIRContext *context, FailurePropagationMode val);
  FailurePropagationMode getValue() const;
};
} // namespace transform
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::transform::FailurePropagationMode, ::mlir::transform::FailurePropagationMode> {
  template <typename ParserT>
  static FailureOr<::mlir::transform::FailurePropagationMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Silenceable error propagation policy");

    // Symbolize the keyword.
    if (::std::optional<::mlir::transform::FailurePropagationMode> attr = ::mlir::transform::symbolizeEnum<::mlir::transform::FailurePropagationMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Silenceable error propagation policy specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::transform::FailurePropagationMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::transform::FailurePropagationMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::transform::FailurePropagationMode getEmptyKey() {
    return static_cast<::mlir::transform::FailurePropagationMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::transform::FailurePropagationMode getTombstoneKey() {
    return static_cast<::mlir::transform::FailurePropagationMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::transform::FailurePropagationMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::transform::FailurePropagationMode &lhs, const ::mlir::transform::FailurePropagationMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace transform {
// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5
enum class MatchCmpIPredicate : uint32_t {
  eq = 0,
  ne = 1,
  lt = 2,
  le = 3,
  gt = 4,
  ge = 5,
};

::std::optional<MatchCmpIPredicate> symbolizeMatchCmpIPredicate(uint32_t);
::llvm::StringRef stringifyMatchCmpIPredicate(MatchCmpIPredicate);
::std::optional<MatchCmpIPredicate> symbolizeMatchCmpIPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMatchCmpIPredicate() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(MatchCmpIPredicate enumValue) {
  return stringifyMatchCmpIPredicate(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MatchCmpIPredicate> symbolizeEnum<MatchCmpIPredicate>(::llvm::StringRef str) {
  return symbolizeMatchCmpIPredicate(str);
}

class MatchCmpIPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MatchCmpIPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MatchCmpIPredicateAttr get(::mlir::MLIRContext *context, MatchCmpIPredicate val);
  MatchCmpIPredicate getValue() const;
};
} // namespace transform
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::transform::MatchCmpIPredicate, ::mlir::transform::MatchCmpIPredicate> {
  template <typename ParserT>
  static FailureOr<::mlir::transform::MatchCmpIPredicate> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5");

    // Symbolize the keyword.
    if (::std::optional<::mlir::transform::MatchCmpIPredicate> attr = ::mlir::transform::symbolizeEnum<::mlir::transform::MatchCmpIPredicate>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5 specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::transform::MatchCmpIPredicate value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::transform::MatchCmpIPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::transform::MatchCmpIPredicate getEmptyKey() {
    return static_cast<::mlir::transform::MatchCmpIPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::transform::MatchCmpIPredicate getTombstoneKey() {
    return static_cast<::mlir::transform::MatchCmpIPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::transform::MatchCmpIPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::transform::MatchCmpIPredicate &lhs, const ::mlir::transform::MatchCmpIPredicate &rhs) {
    return lhs == rhs;
  }
};
}

