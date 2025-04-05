/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OpenMPOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
// CancellationConstructType Clause
enum class ClauseCancellationConstructType : uint32_t {
  Parallel = 0,
  Loop = 1,
  Sections = 2,
  Taskgroup = 3,
};

::std::optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(uint32_t);
::llvm::StringRef stringifyClauseCancellationConstructType(ClauseCancellationConstructType);
::std::optional<ClauseCancellationConstructType> symbolizeClauseCancellationConstructType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseCancellationConstructType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ClauseCancellationConstructType enumValue) {
  return stringifyClauseCancellationConstructType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseCancellationConstructType> symbolizeEnum<ClauseCancellationConstructType>(::llvm::StringRef str) {
  return symbolizeClauseCancellationConstructType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseCancellationConstructType, ::mlir::omp::ClauseCancellationConstructType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseCancellationConstructType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for CancellationConstructType Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseCancellationConstructType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseCancellationConstructType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid CancellationConstructType Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseCancellationConstructType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseCancellationConstructType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseCancellationConstructType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseCancellationConstructType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseCancellationConstructType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseCancellationConstructType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseCancellationConstructType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseCancellationConstructType &lhs, const ::mlir::omp::ClauseCancellationConstructType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// depend clause
enum class ClauseDepend : uint32_t {
  dependsource = 0,
  dependsink = 1,
};

::std::optional<ClauseDepend> symbolizeClauseDepend(uint32_t);
::llvm::StringRef stringifyClauseDepend(ClauseDepend);
::std::optional<ClauseDepend> symbolizeClauseDepend(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseDepend() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseDepend enumValue) {
  return stringifyClauseDepend(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseDepend> symbolizeEnum<ClauseDepend>(::llvm::StringRef str) {
  return symbolizeClauseDepend(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseDepend, ::mlir::omp::ClauseDepend> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseDepend> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for depend clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseDepend> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseDepend>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid depend clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseDepend value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseDepend> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseDepend getEmptyKey() {
    return static_cast<::mlir::omp::ClauseDepend>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseDepend getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseDepend>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseDepend &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseDepend &lhs, const ::mlir::omp::ClauseDepend &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// requires clauses
enum class ClauseRequires : uint32_t {
  none = 0,
  reverse_offload = 1,
  unified_address = 2,
  unified_shared_memory = 4,
  dynamic_allocators = 8,
};

::std::optional<ClauseRequires> symbolizeClauseRequires(uint32_t);
std::string stringifyClauseRequires(ClauseRequires);
::std::optional<ClauseRequires> symbolizeClauseRequires(::llvm::StringRef);

inline constexpr ClauseRequires operator|(ClauseRequires a, ClauseRequires b) {
  return static_cast<ClauseRequires>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr ClauseRequires operator&(ClauseRequires a, ClauseRequires b) {
  return static_cast<ClauseRequires>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr ClauseRequires operator^(ClauseRequires a, ClauseRequires b) {
  return static_cast<ClauseRequires>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr ClauseRequires operator~(ClauseRequires bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<ClauseRequires>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(15u));
}
inline constexpr bool bitEnumContainsAll(ClauseRequires bits, ClauseRequires bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(ClauseRequires bits, ClauseRequires bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr ClauseRequires bitEnumClear(ClauseRequires bits, ClauseRequires bit) {
  return bits & ~bit;
}
inline constexpr ClauseRequires bitEnumSet(ClauseRequires bits, ClauseRequires bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(ClauseRequires enumValue) {
  return stringifyClauseRequires(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseRequires> symbolizeEnum<ClauseRequires>(::llvm::StringRef str) {
  return symbolizeClauseRequires(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseRequires, ::mlir::omp::ClauseRequires> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseRequires> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for requires clauses");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseRequires> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseRequires>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid requires clauses specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseRequires value) {
  auto valueStr = stringifyEnum(value);
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::omp::ClauseRequires>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseRequires> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseRequires getEmptyKey() {
    return static_cast<::mlir::omp::ClauseRequires>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseRequires getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseRequires>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseRequires &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseRequires &lhs, const ::mlir::omp::ClauseRequires &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// depend clause in a target or task construct
enum class ClauseTaskDepend : uint32_t {
  taskdependin = 0,
  taskdependout = 1,
  taskdependinout = 2,
};

::std::optional<ClauseTaskDepend> symbolizeClauseTaskDepend(uint32_t);
::llvm::StringRef stringifyClauseTaskDepend(ClauseTaskDepend);
::std::optional<ClauseTaskDepend> symbolizeClauseTaskDepend(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseTaskDepend() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(ClauseTaskDepend enumValue) {
  return stringifyClauseTaskDepend(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseTaskDepend> symbolizeEnum<ClauseTaskDepend>(::llvm::StringRef str) {
  return symbolizeClauseTaskDepend(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseTaskDepend, ::mlir::omp::ClauseTaskDepend> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseTaskDepend> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for depend clause in a target or task construct");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseTaskDepend> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseTaskDepend>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid depend clause in a target or task construct specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseTaskDepend value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseTaskDepend> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseTaskDepend getEmptyKey() {
    return static_cast<::mlir::omp::ClauseTaskDepend>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseTaskDepend getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseTaskDepend>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseTaskDepend &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseTaskDepend &lhs, const ::mlir::omp::ClauseTaskDepend &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// Type of a data-sharing clause
enum class DataSharingClauseType : uint32_t {
  Private = 0,
  FirstPrivate = 1,
};

::std::optional<DataSharingClauseType> symbolizeDataSharingClauseType(uint32_t);
::llvm::StringRef stringifyDataSharingClauseType(DataSharingClauseType);
::std::optional<DataSharingClauseType> symbolizeDataSharingClauseType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDataSharingClauseType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(DataSharingClauseType enumValue) {
  return stringifyDataSharingClauseType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DataSharingClauseType> symbolizeEnum<DataSharingClauseType>(::llvm::StringRef str) {
  return symbolizeDataSharingClauseType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::DataSharingClauseType, ::mlir::omp::DataSharingClauseType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::DataSharingClauseType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Type of a data-sharing clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::DataSharingClauseType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::DataSharingClauseType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Type of a data-sharing clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::DataSharingClauseType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::DataSharingClauseType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::DataSharingClauseType getEmptyKey() {
    return static_cast<::mlir::omp::DataSharingClauseType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::DataSharingClauseType getTombstoneKey() {
    return static_cast<::mlir::omp::DataSharingClauseType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::DataSharingClauseType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::DataSharingClauseType &lhs, const ::mlir::omp::DataSharingClauseType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// capture clause
enum class DeclareTargetCaptureClause : uint32_t {
  to = 0,
  link = 1,
  enter = 2,
};

::std::optional<DeclareTargetCaptureClause> symbolizeDeclareTargetCaptureClause(uint32_t);
::llvm::StringRef stringifyDeclareTargetCaptureClause(DeclareTargetCaptureClause);
::std::optional<DeclareTargetCaptureClause> symbolizeDeclareTargetCaptureClause(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDeclareTargetCaptureClause() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(DeclareTargetCaptureClause enumValue) {
  return stringifyDeclareTargetCaptureClause(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DeclareTargetCaptureClause> symbolizeEnum<DeclareTargetCaptureClause>(::llvm::StringRef str) {
  return symbolizeDeclareTargetCaptureClause(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::DeclareTargetCaptureClause, ::mlir::omp::DeclareTargetCaptureClause> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::DeclareTargetCaptureClause> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for capture clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::DeclareTargetCaptureClause> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::DeclareTargetCaptureClause>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid capture clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::DeclareTargetCaptureClause value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::DeclareTargetCaptureClause> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::DeclareTargetCaptureClause getEmptyKey() {
    return static_cast<::mlir::omp::DeclareTargetCaptureClause>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::DeclareTargetCaptureClause getTombstoneKey() {
    return static_cast<::mlir::omp::DeclareTargetCaptureClause>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::DeclareTargetCaptureClause &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::DeclareTargetCaptureClause &lhs, const ::mlir::omp::DeclareTargetCaptureClause &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// device_type clause
enum class DeclareTargetDeviceType : uint32_t {
  any = 0,
  host = 1,
  nohost = 2,
};

::std::optional<DeclareTargetDeviceType> symbolizeDeclareTargetDeviceType(uint32_t);
::llvm::StringRef stringifyDeclareTargetDeviceType(DeclareTargetDeviceType);
::std::optional<DeclareTargetDeviceType> symbolizeDeclareTargetDeviceType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDeclareTargetDeviceType() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(DeclareTargetDeviceType enumValue) {
  return stringifyDeclareTargetDeviceType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DeclareTargetDeviceType> symbolizeEnum<DeclareTargetDeviceType>(::llvm::StringRef str) {
  return symbolizeDeclareTargetDeviceType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::DeclareTargetDeviceType, ::mlir::omp::DeclareTargetDeviceType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::DeclareTargetDeviceType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for device_type clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::DeclareTargetDeviceType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::DeclareTargetDeviceType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid device_type clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::DeclareTargetDeviceType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::DeclareTargetDeviceType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::DeclareTargetDeviceType getEmptyKey() {
    return static_cast<::mlir::omp::DeclareTargetDeviceType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::DeclareTargetDeviceType getTombstoneKey() {
    return static_cast<::mlir::omp::DeclareTargetDeviceType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::DeclareTargetDeviceType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::DeclareTargetDeviceType &lhs, const ::mlir::omp::DeclareTargetDeviceType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// GrainsizeType Clause
enum class ClauseGrainsizeType : uint32_t {
  Strict = 0,
};

::std::optional<ClauseGrainsizeType> symbolizeClauseGrainsizeType(uint32_t);
::llvm::StringRef stringifyClauseGrainsizeType(ClauseGrainsizeType);
::std::optional<ClauseGrainsizeType> symbolizeClauseGrainsizeType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseGrainsizeType() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(ClauseGrainsizeType enumValue) {
  return stringifyClauseGrainsizeType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseGrainsizeType> symbolizeEnum<ClauseGrainsizeType>(::llvm::StringRef str) {
  return symbolizeClauseGrainsizeType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseGrainsizeType, ::mlir::omp::ClauseGrainsizeType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseGrainsizeType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for GrainsizeType Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseGrainsizeType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseGrainsizeType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid GrainsizeType Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseGrainsizeType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseGrainsizeType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseGrainsizeType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseGrainsizeType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseGrainsizeType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseGrainsizeType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseGrainsizeType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseGrainsizeType &lhs, const ::mlir::omp::ClauseGrainsizeType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// MemoryOrderKind Clause
enum class ClauseMemoryOrderKind : uint32_t {
  Seq_cst = 0,
  Acq_rel = 1,
  Acquire = 2,
  Release = 3,
  Relaxed = 4,
};

::std::optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(uint32_t);
::llvm::StringRef stringifyClauseMemoryOrderKind(ClauseMemoryOrderKind);
::std::optional<ClauseMemoryOrderKind> symbolizeClauseMemoryOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseMemoryOrderKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ClauseMemoryOrderKind enumValue) {
  return stringifyClauseMemoryOrderKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseMemoryOrderKind> symbolizeEnum<ClauseMemoryOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseMemoryOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseMemoryOrderKind, ::mlir::omp::ClauseMemoryOrderKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseMemoryOrderKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for MemoryOrderKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseMemoryOrderKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseMemoryOrderKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid MemoryOrderKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseMemoryOrderKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseMemoryOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseMemoryOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseMemoryOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseMemoryOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseMemoryOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseMemoryOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseMemoryOrderKind &lhs, const ::mlir::omp::ClauseMemoryOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// NumTasksType Clause
enum class ClauseNumTasksType : uint32_t {
  Strict = 0,
};

::std::optional<ClauseNumTasksType> symbolizeClauseNumTasksType(uint32_t);
::llvm::StringRef stringifyClauseNumTasksType(ClauseNumTasksType);
::std::optional<ClauseNumTasksType> symbolizeClauseNumTasksType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseNumTasksType() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(ClauseNumTasksType enumValue) {
  return stringifyClauseNumTasksType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseNumTasksType> symbolizeEnum<ClauseNumTasksType>(::llvm::StringRef str) {
  return symbolizeClauseNumTasksType(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseNumTasksType, ::mlir::omp::ClauseNumTasksType> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseNumTasksType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for NumTasksType Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseNumTasksType> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseNumTasksType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid NumTasksType Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseNumTasksType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseNumTasksType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseNumTasksType getEmptyKey() {
    return static_cast<::mlir::omp::ClauseNumTasksType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseNumTasksType getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseNumTasksType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseNumTasksType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseNumTasksType &lhs, const ::mlir::omp::ClauseNumTasksType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OrderKind Clause
enum class ClauseOrderKind : uint32_t {
  Concurrent = 1,
};

::std::optional<ClauseOrderKind> symbolizeClauseOrderKind(uint32_t);
::llvm::StringRef stringifyClauseOrderKind(ClauseOrderKind);
::std::optional<ClauseOrderKind> symbolizeClauseOrderKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseOrderKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseOrderKind enumValue) {
  return stringifyClauseOrderKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseOrderKind> symbolizeEnum<ClauseOrderKind>(::llvm::StringRef str) {
  return symbolizeClauseOrderKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseOrderKind, ::mlir::omp::ClauseOrderKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseOrderKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for OrderKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseOrderKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseOrderKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid OrderKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseOrderKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseOrderKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseOrderKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseOrderKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseOrderKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseOrderKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseOrderKind &lhs, const ::mlir::omp::ClauseOrderKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OpenMP Order Modifier
enum class OrderModifier : uint32_t {
  reproducible = 0,
  unconstrained = 1,
};

::std::optional<OrderModifier> symbolizeOrderModifier(uint32_t);
::llvm::StringRef stringifyOrderModifier(OrderModifier);
::std::optional<OrderModifier> symbolizeOrderModifier(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForOrderModifier() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(OrderModifier enumValue) {
  return stringifyOrderModifier(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<OrderModifier> symbolizeEnum<OrderModifier>(::llvm::StringRef str) {
  return symbolizeOrderModifier(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::OrderModifier, ::mlir::omp::OrderModifier> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::OrderModifier> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for OpenMP Order Modifier");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::OrderModifier> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::OrderModifier>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid OpenMP Order Modifier specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::OrderModifier value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::OrderModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::OrderModifier getEmptyKey() {
    return static_cast<::mlir::omp::OrderModifier>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::OrderModifier getTombstoneKey() {
    return static_cast<::mlir::omp::OrderModifier>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::OrderModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::OrderModifier &lhs, const ::mlir::omp::OrderModifier &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ProcBindKind Clause
enum class ClauseProcBindKind : uint32_t {
  Primary = 0,
  Master = 1,
  Close = 2,
  Spread = 3,
};

::std::optional<ClauseProcBindKind> symbolizeClauseProcBindKind(uint32_t);
::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind);
::std::optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseProcBindKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ClauseProcBindKind enumValue) {
  return stringifyClauseProcBindKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseProcBindKind> symbolizeEnum<ClauseProcBindKind>(::llvm::StringRef str) {
  return symbolizeClauseProcBindKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseProcBindKind, ::mlir::omp::ClauseProcBindKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseProcBindKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ProcBindKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseProcBindKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseProcBindKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ProcBindKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseProcBindKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseProcBindKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseProcBindKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseProcBindKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseProcBindKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseProcBindKind &lhs, const ::mlir::omp::ClauseProcBindKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ScheduleKind Clause
enum class ClauseScheduleKind : uint32_t {
  Static = 0,
  Dynamic = 1,
  Guided = 2,
  Auto = 3,
  Runtime = 4,
};

::std::optional<ClauseScheduleKind> symbolizeClauseScheduleKind(uint32_t);
::llvm::StringRef stringifyClauseScheduleKind(ClauseScheduleKind);
::std::optional<ClauseScheduleKind> symbolizeClauseScheduleKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseScheduleKind() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ClauseScheduleKind enumValue) {
  return stringifyClauseScheduleKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseScheduleKind> symbolizeEnum<ClauseScheduleKind>(::llvm::StringRef str) {
  return symbolizeClauseScheduleKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ClauseScheduleKind, ::mlir::omp::ClauseScheduleKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ClauseScheduleKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ScheduleKind Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ClauseScheduleKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ClauseScheduleKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ScheduleKind Clause specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ClauseScheduleKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseScheduleKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ClauseScheduleKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseScheduleKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseScheduleKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseScheduleKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseScheduleKind &lhs, const ::mlir::omp::ClauseScheduleKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// OpenMP Schedule Modifier
enum class ScheduleModifier : uint32_t {
  none = 0,
  monotonic = 1,
  nonmonotonic = 2,
  simd = 3,
};

::std::optional<ScheduleModifier> symbolizeScheduleModifier(uint32_t);
::llvm::StringRef stringifyScheduleModifier(ScheduleModifier);
::std::optional<ScheduleModifier> symbolizeScheduleModifier(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForScheduleModifier() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ScheduleModifier enumValue) {
  return stringifyScheduleModifier(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ScheduleModifier> symbolizeEnum<ScheduleModifier>(::llvm::StringRef str) {
  return symbolizeScheduleModifier(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::ScheduleModifier, ::mlir::omp::ScheduleModifier> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::ScheduleModifier> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for OpenMP Schedule Modifier");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::ScheduleModifier> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::ScheduleModifier>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid OpenMP Schedule Modifier specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::ScheduleModifier value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ScheduleModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::ScheduleModifier getEmptyKey() {
    return static_cast<::mlir::omp::ScheduleModifier>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ScheduleModifier getTombstoneKey() {
    return static_cast<::mlir::omp::ScheduleModifier>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ScheduleModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::ScheduleModifier &lhs, const ::mlir::omp::ScheduleModifier &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// variable capture kind
enum class VariableCaptureKind : uint32_t {
  This = 0,
  ByRef = 1,
  ByCopy = 2,
  VLAType = 3,
};

::std::optional<VariableCaptureKind> symbolizeVariableCaptureKind(uint32_t);
::llvm::StringRef stringifyVariableCaptureKind(VariableCaptureKind);
::std::optional<VariableCaptureKind> symbolizeVariableCaptureKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVariableCaptureKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(VariableCaptureKind enumValue) {
  return stringifyVariableCaptureKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<VariableCaptureKind> symbolizeEnum<VariableCaptureKind>(::llvm::StringRef str) {
  return symbolizeVariableCaptureKind(str);
}
} // namespace omp
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::omp::VariableCaptureKind, ::mlir::omp::VariableCaptureKind> {
  template <typename ParserT>
  static FailureOr<::mlir::omp::VariableCaptureKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for variable capture kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::omp::VariableCaptureKind> attr = ::mlir::omp::symbolizeEnum<::mlir::omp::VariableCaptureKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid variable capture kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::omp::VariableCaptureKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::VariableCaptureKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::omp::VariableCaptureKind getEmptyKey() {
    return static_cast<::mlir::omp::VariableCaptureKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::VariableCaptureKind getTombstoneKey() {
    return static_cast<::mlir::omp::VariableCaptureKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::VariableCaptureKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::omp::VariableCaptureKind &lhs, const ::mlir::omp::VariableCaptureKind &rhs) {
    return lhs == rhs;
  }
};
}

