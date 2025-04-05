/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: IndexEnums.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace index {
// index comparison predicate kind
enum class IndexCmpPredicate : uint32_t {
  EQ = 0,
  NE = 1,
  SLT = 2,
  SLE = 3,
  SGT = 4,
  SGE = 5,
  ULT = 6,
  ULE = 7,
  UGT = 8,
  UGE = 9,
};

::std::optional<IndexCmpPredicate> symbolizeIndexCmpPredicate(uint32_t);
::llvm::StringRef stringifyIndexCmpPredicate(IndexCmpPredicate);
::std::optional<IndexCmpPredicate> symbolizeIndexCmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForIndexCmpPredicate() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(IndexCmpPredicate enumValue) {
  return stringifyIndexCmpPredicate(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<IndexCmpPredicate> symbolizeEnum<IndexCmpPredicate>(::llvm::StringRef str) {
  return symbolizeIndexCmpPredicate(str);
}
} // namespace index
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::index::IndexCmpPredicate, ::mlir::index::IndexCmpPredicate> {
  template <typename ParserT>
  static FailureOr<::mlir::index::IndexCmpPredicate> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for index comparison predicate kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::index::IndexCmpPredicate> attr = ::mlir::index::symbolizeEnum<::mlir::index::IndexCmpPredicate>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid index comparison predicate kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::index::IndexCmpPredicate value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::index::IndexCmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::index::IndexCmpPredicate getEmptyKey() {
    return static_cast<::mlir::index::IndexCmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::index::IndexCmpPredicate getTombstoneKey() {
    return static_cast<::mlir::index::IndexCmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::index::IndexCmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::index::IndexCmpPredicate &lhs, const ::mlir::index::IndexCmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

