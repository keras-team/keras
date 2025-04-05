/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace NVVM {
// NVVM load cache modifier kind
enum class LoadCacheModifierKind : uint32_t {
  CA = 0,
  CG = 1,
  CS = 2,
  LU = 3,
  CV = 4,
};

::std::optional<LoadCacheModifierKind> symbolizeLoadCacheModifierKind(uint32_t);
::llvm::StringRef stringifyLoadCacheModifierKind(LoadCacheModifierKind);
::std::optional<LoadCacheModifierKind> symbolizeLoadCacheModifierKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLoadCacheModifierKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(LoadCacheModifierKind enumValue) {
  return stringifyLoadCacheModifierKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<LoadCacheModifierKind> symbolizeEnum<LoadCacheModifierKind>(::llvm::StringRef str) {
  return symbolizeLoadCacheModifierKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::LoadCacheModifierKind, ::mlir::NVVM::LoadCacheModifierKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::LoadCacheModifierKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM load cache modifier kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::LoadCacheModifierKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::LoadCacheModifierKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM load cache modifier kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::LoadCacheModifierKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::LoadCacheModifierKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::LoadCacheModifierKind getEmptyKey() {
    return static_cast<::mlir::NVVM::LoadCacheModifierKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::LoadCacheModifierKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::LoadCacheModifierKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::LoadCacheModifierKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::LoadCacheModifierKind &lhs, const ::mlir::NVVM::LoadCacheModifierKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// MMA binary operations
enum class MMAB1Op : uint32_t {
  none = 0,
  xor_popc = 1,
  and_popc = 2,
};

::std::optional<MMAB1Op> symbolizeMMAB1Op(uint32_t);
::llvm::StringRef stringifyMMAB1Op(MMAB1Op);
::std::optional<MMAB1Op> symbolizeMMAB1Op(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAB1Op() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(MMAB1Op enumValue) {
  return stringifyMMAB1Op(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAB1Op> symbolizeEnum<MMAB1Op>(::llvm::StringRef str) {
  return symbolizeMMAB1Op(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMAB1Op, ::mlir::NVVM::MMAB1Op> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMAB1Op> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for MMA binary operations");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMAB1Op> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMAB1Op>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid MMA binary operations specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMAB1Op value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAB1Op> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAB1Op getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAB1Op>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAB1Op getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAB1Op>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAB1Op &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAB1Op &lhs, const ::mlir::NVVM::MMAB1Op &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA frag type
enum class MMAFrag : uint32_t {
  a = 0,
  b = 1,
  c = 2,
};

::std::optional<MMAFrag> symbolizeMMAFrag(uint32_t);
::llvm::StringRef stringifyMMAFrag(MMAFrag);
::std::optional<MMAFrag> symbolizeMMAFrag(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAFrag() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(MMAFrag enumValue) {
  return stringifyMMAFrag(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAFrag> symbolizeEnum<MMAFrag>(::llvm::StringRef str) {
  return symbolizeMMAFrag(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMAFrag, ::mlir::NVVM::MMAFrag> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMAFrag> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM MMA frag type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMAFrag> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMAFrag>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM MMA frag type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMAFrag value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAFrag> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAFrag getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAFrag>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAFrag getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAFrag>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAFrag &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAFrag &lhs, const ::mlir::NVVM::MMAFrag &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// MMA overflow options
enum class MMAIntOverflow : uint32_t {
  satfinite = 1,
  wrapped = 0,
};

::std::optional<MMAIntOverflow> symbolizeMMAIntOverflow(uint32_t);
::llvm::StringRef stringifyMMAIntOverflow(MMAIntOverflow);
::std::optional<MMAIntOverflow> symbolizeMMAIntOverflow(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAIntOverflow() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MMAIntOverflow enumValue) {
  return stringifyMMAIntOverflow(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAIntOverflow> symbolizeEnum<MMAIntOverflow>(::llvm::StringRef str) {
  return symbolizeMMAIntOverflow(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMAIntOverflow, ::mlir::NVVM::MMAIntOverflow> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMAIntOverflow> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for MMA overflow options");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMAIntOverflow> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMAIntOverflow>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid MMA overflow options specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMAIntOverflow value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMAIntOverflow> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMAIntOverflow getEmptyKey() {
    return static_cast<::mlir::NVVM::MMAIntOverflow>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMAIntOverflow getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMAIntOverflow>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMAIntOverflow &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMAIntOverflow &lhs, const ::mlir::NVVM::MMAIntOverflow &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA layout
enum class MMALayout : uint32_t {
  row = 0,
  col = 1,
};

::std::optional<MMALayout> symbolizeMMALayout(uint32_t);
::llvm::StringRef stringifyMMALayout(MMALayout);
::std::optional<MMALayout> symbolizeMMALayout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMALayout() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MMALayout enumValue) {
  return stringifyMMALayout(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMALayout> symbolizeEnum<MMALayout>(::llvm::StringRef str) {
  return symbolizeMMALayout(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMALayout, ::mlir::NVVM::MMALayout> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMALayout> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM MMA layout");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMALayout> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMALayout>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM MMA layout specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMALayout value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMALayout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMALayout getEmptyKey() {
    return static_cast<::mlir::NVVM::MMALayout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMALayout getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMALayout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMALayout &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMALayout &lhs, const ::mlir::NVVM::MMALayout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM MMA types
enum class MMATypes : uint32_t {
  f16 = 0,
  f32 = 1,
  tf32 = 2,
  bf16 = 9,
  s8 = 4,
  u8 = 3,
  s32 = 5,
  s4 = 8,
  u4 = 7,
  b1 = 6,
  f64 = 10,
};

::std::optional<MMATypes> symbolizeMMATypes(uint32_t);
::llvm::StringRef stringifyMMATypes(MMATypes);
::std::optional<MMATypes> symbolizeMMATypes(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMATypes() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(MMATypes enumValue) {
  return stringifyMMATypes(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMATypes> symbolizeEnum<MMATypes>(::llvm::StringRef str) {
  return symbolizeMMATypes(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MMATypes, ::mlir::NVVM::MMATypes> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MMATypes> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM MMA types");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MMATypes> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MMATypes>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM MMA types specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MMATypes value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MMATypes> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MMATypes getEmptyKey() {
    return static_cast<::mlir::NVVM::MMATypes>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MMATypes getTombstoneKey() {
    return static_cast<::mlir::NVVM::MMATypes>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MMATypes &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MMATypes &lhs, const ::mlir::NVVM::MMATypes &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM Memory Scope kind
enum class MemScopeKind : uint32_t {
  CTA = 0,
  CLUSTER = 1,
  GPU = 2,
  SYS = 3,
};

::std::optional<MemScopeKind> symbolizeMemScopeKind(uint32_t);
::llvm::StringRef stringifyMemScopeKind(MemScopeKind);
::std::optional<MemScopeKind> symbolizeMemScopeKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemScopeKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(MemScopeKind enumValue) {
  return stringifyMemScopeKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MemScopeKind> symbolizeEnum<MemScopeKind>(::llvm::StringRef str) {
  return symbolizeMemScopeKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::MemScopeKind, ::mlir::NVVM::MemScopeKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::MemScopeKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM Memory Scope kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::MemScopeKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::MemScopeKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM Memory Scope kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::MemScopeKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::MemScopeKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::MemScopeKind getEmptyKey() {
    return static_cast<::mlir::NVVM::MemScopeKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::MemScopeKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::MemScopeKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::MemScopeKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::MemScopeKind &lhs, const ::mlir::NVVM::MemScopeKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// Proxy kind
enum class ProxyKind : uint32_t {
  alias = 0,
  async = 1,
  async_global = 2,
  async_shared = 3,
  TENSORMAP = 4,
  GENERIC = 5,
};

::std::optional<ProxyKind> symbolizeProxyKind(uint32_t);
::llvm::StringRef stringifyProxyKind(ProxyKind);
::std::optional<ProxyKind> symbolizeProxyKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForProxyKind() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(ProxyKind enumValue) {
  return stringifyProxyKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ProxyKind> symbolizeEnum<ProxyKind>(::llvm::StringRef str) {
  return symbolizeProxyKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::ProxyKind, ::mlir::NVVM::ProxyKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::ProxyKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Proxy kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::ProxyKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::ProxyKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Proxy kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::ProxyKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::ProxyKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::ProxyKind getEmptyKey() {
    return static_cast<::mlir::NVVM::ProxyKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::ProxyKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::ProxyKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::ProxyKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::ProxyKind &lhs, const ::mlir::NVVM::ProxyKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM redux kind
enum class ReduxKind : uint32_t {
  ADD = 1,
  AND = 2,
  MAX = 3,
  MIN = 4,
  OR = 5,
  UMAX = 6,
  UMIN = 7,
  XOR = 8,
};

::std::optional<ReduxKind> symbolizeReduxKind(uint32_t);
::llvm::StringRef stringifyReduxKind(ReduxKind);
::std::optional<ReduxKind> symbolizeReduxKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReduxKind() {
  return 8;
}


inline ::llvm::StringRef stringifyEnum(ReduxKind enumValue) {
  return stringifyReduxKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReduxKind> symbolizeEnum<ReduxKind>(::llvm::StringRef str) {
  return symbolizeReduxKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::ReduxKind, ::mlir::NVVM::ReduxKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::ReduxKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM redux kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::ReduxKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::ReduxKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM redux kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::ReduxKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::ReduxKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::ReduxKind getEmptyKey() {
    return static_cast<::mlir::NVVM::ReduxKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::ReduxKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::ReduxKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::ReduxKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::ReduxKind &lhs, const ::mlir::NVVM::ReduxKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM set max register action
enum class SetMaxRegisterAction : uint32_t {
  decrease = 1,
  increase = 0,
};

::std::optional<SetMaxRegisterAction> symbolizeSetMaxRegisterAction(uint32_t);
::llvm::StringRef stringifySetMaxRegisterAction(SetMaxRegisterAction);
::std::optional<SetMaxRegisterAction> symbolizeSetMaxRegisterAction(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSetMaxRegisterAction() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(SetMaxRegisterAction enumValue) {
  return stringifySetMaxRegisterAction(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SetMaxRegisterAction> symbolizeEnum<SetMaxRegisterAction>(::llvm::StringRef str) {
  return symbolizeSetMaxRegisterAction(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::SetMaxRegisterAction, ::mlir::NVVM::SetMaxRegisterAction> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::SetMaxRegisterAction> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM set max register action");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::SetMaxRegisterAction> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::SetMaxRegisterAction>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM set max register action specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::SetMaxRegisterAction value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::SetMaxRegisterAction> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::SetMaxRegisterAction getEmptyKey() {
    return static_cast<::mlir::NVVM::SetMaxRegisterAction>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::SetMaxRegisterAction getTombstoneKey() {
    return static_cast<::mlir::NVVM::SetMaxRegisterAction>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::SetMaxRegisterAction &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::SetMaxRegisterAction &lhs, const ::mlir::NVVM::SetMaxRegisterAction &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// Shared memory space
enum class SharedSpace : uint32_t {
  shared_cta = 0,
  shared_cluster = 1,
};

::std::optional<SharedSpace> symbolizeSharedSpace(uint32_t);
::llvm::StringRef stringifySharedSpace(SharedSpace);
::std::optional<SharedSpace> symbolizeSharedSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSharedSpace() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(SharedSpace enumValue) {
  return stringifySharedSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SharedSpace> symbolizeEnum<SharedSpace>(::llvm::StringRef str) {
  return symbolizeSharedSpace(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::SharedSpace, ::mlir::NVVM::SharedSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::SharedSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Shared memory space");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::SharedSpace> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::SharedSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Shared memory space specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::SharedSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::SharedSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::SharedSpace getEmptyKey() {
    return static_cast<::mlir::NVVM::SharedSpace>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::SharedSpace getTombstoneKey() {
    return static_cast<::mlir::NVVM::SharedSpace>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::SharedSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::SharedSpace &lhs, const ::mlir::NVVM::SharedSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM shuffle kind
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};

::std::optional<ShflKind> symbolizeShflKind(uint32_t);
::llvm::StringRef stringifyShflKind(ShflKind);
::std::optional<ShflKind> symbolizeShflKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForShflKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ShflKind enumValue) {
  return stringifyShflKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ShflKind> symbolizeEnum<ShflKind>(::llvm::StringRef str) {
  return symbolizeShflKind(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::ShflKind, ::mlir::NVVM::ShflKind> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::ShflKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM shuffle kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::ShflKind> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::ShflKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM shuffle kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::ShflKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::ShflKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::ShflKind getEmptyKey() {
    return static_cast<::mlir::NVVM::ShflKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::ShflKind getTombstoneKey() {
    return static_cast<::mlir::NVVM::ShflKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::ShflKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::ShflKind &lhs, const ::mlir::NVVM::ShflKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// WGMMA overflow options
enum class WGMMAScaleIn : uint32_t {
  one = 1,
  neg,
};

::llvm::StringRef stringifyWGMMAScaleIn(WGMMAScaleIn);
::std::optional<WGMMAScaleIn> symbolizeWGMMAScaleIn(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(WGMMAScaleIn enumValue) {
  return stringifyWGMMAScaleIn(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<WGMMAScaleIn> symbolizeEnum<WGMMAScaleIn>(::llvm::StringRef str) {
  return symbolizeWGMMAScaleIn(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::WGMMAScaleIn, ::mlir::NVVM::WGMMAScaleIn> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::WGMMAScaleIn> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for WGMMA overflow options");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::WGMMAScaleIn> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::WGMMAScaleIn>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid WGMMA overflow options specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::WGMMAScaleIn value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::WGMMAScaleIn> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::WGMMAScaleIn getEmptyKey() {
    return static_cast<::mlir::NVVM::WGMMAScaleIn>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::WGMMAScaleIn getTombstoneKey() {
    return static_cast<::mlir::NVVM::WGMMAScaleIn>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::WGMMAScaleIn &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::WGMMAScaleIn &lhs, const ::mlir::NVVM::WGMMAScaleIn &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// WGMMA input predicate
enum class WGMMAScaleOut : uint32_t {
  zero = 0,
  one = 1,
};

::std::optional<WGMMAScaleOut> symbolizeWGMMAScaleOut(uint32_t);
::llvm::StringRef stringifyWGMMAScaleOut(WGMMAScaleOut);
::std::optional<WGMMAScaleOut> symbolizeWGMMAScaleOut(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForWGMMAScaleOut() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(WGMMAScaleOut enumValue) {
  return stringifyWGMMAScaleOut(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<WGMMAScaleOut> symbolizeEnum<WGMMAScaleOut>(::llvm::StringRef str) {
  return symbolizeWGMMAScaleOut(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::WGMMAScaleOut, ::mlir::NVVM::WGMMAScaleOut> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::WGMMAScaleOut> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for WGMMA input predicate");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::WGMMAScaleOut> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::WGMMAScaleOut>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid WGMMA input predicate specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::WGMMAScaleOut value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::WGMMAScaleOut> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::WGMMAScaleOut getEmptyKey() {
    return static_cast<::mlir::NVVM::WGMMAScaleOut>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::WGMMAScaleOut getTombstoneKey() {
    return static_cast<::mlir::NVVM::WGMMAScaleOut>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::WGMMAScaleOut &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::WGMMAScaleOut &lhs, const ::mlir::NVVM::WGMMAScaleOut &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace NVVM {
// NVVM WGMMA types
enum class WGMMATypes : uint32_t {
  f16 = 0,
  tf32 = 1,
  u8 = 2,
  s8 = 3,
  b1 = 4,
  bf16 = 5,
  e4m3 = 6,
  e5m2 = 7,
  f32 = 8,
  s32 = 9,
};

::std::optional<WGMMATypes> symbolizeWGMMATypes(uint32_t);
::llvm::StringRef stringifyWGMMATypes(WGMMATypes);
::std::optional<WGMMATypes> symbolizeWGMMATypes(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForWGMMATypes() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(WGMMATypes enumValue) {
  return stringifyWGMMATypes(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<WGMMATypes> symbolizeEnum<WGMMATypes>(::llvm::StringRef str) {
  return symbolizeWGMMATypes(str);
}
} // namespace NVVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::NVVM::WGMMATypes, ::mlir::NVVM::WGMMATypes> {
  template <typename ParserT>
  static FailureOr<::mlir::NVVM::WGMMATypes> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NVVM WGMMA types");

    // Symbolize the keyword.
    if (::std::optional<::mlir::NVVM::WGMMATypes> attr = ::mlir::NVVM::symbolizeEnum<::mlir::NVVM::WGMMATypes>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NVVM WGMMA types specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::NVVM::WGMMATypes value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::NVVM::WGMMATypes> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::NVVM::WGMMATypes getEmptyKey() {
    return static_cast<::mlir::NVVM::WGMMATypes>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::NVVM::WGMMATypes getTombstoneKey() {
    return static_cast<::mlir::NVVM::WGMMATypes>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::NVVM::WGMMATypes &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::NVVM::WGMMATypes &lhs, const ::mlir::NVVM::WGMMATypes &rhs) {
    return lhs == rhs;
  }
};
}

