/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MeshOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mesh {
// Reduction of an iterator/mesh dimension.
enum class ReductionKind : uint32_t {
  Sum = 1,
  Max = 2,
  Min = 3,
  Product = 4,
  Average = 5,
  BitwiseAnd = 6,
  BitwiseOr = 7,
  BitwiseXor = 8,
  Generic = 100,
};

::std::optional<ReductionKind> symbolizeReductionKind(uint32_t);
::llvm::StringRef stringifyReductionKind(ReductionKind);
::std::optional<ReductionKind> symbolizeReductionKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReductionKind() {
  return 100;
}


inline ::llvm::StringRef stringifyEnum(ReductionKind enumValue) {
  return stringifyReductionKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReductionKind> symbolizeEnum<ReductionKind>(::llvm::StringRef str) {
  return symbolizeReductionKind(str);
}
} // namespace mesh
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::mesh::ReductionKind, ::mlir::mesh::ReductionKind> {
  template <typename ParserT>
  static FailureOr<::mlir::mesh::ReductionKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Reduction of an iterator/mesh dimension.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::mesh::ReductionKind> attr = ::mlir::mesh::symbolizeEnum<::mlir::mesh::ReductionKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Reduction of an iterator/mesh dimension. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::mesh::ReductionKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::mesh::ReductionKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mesh::ReductionKind getEmptyKey() {
    return static_cast<::mlir::mesh::ReductionKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mesh::ReductionKind getTombstoneKey() {
    return static_cast<::mlir::mesh::ReductionKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mesh::ReductionKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mesh::ReductionKind &lhs, const ::mlir::mesh::ReductionKind &rhs) {
    return lhs == rhs;
  }
};
}

