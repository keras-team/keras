/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: GPUOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
// GPU address space
enum class AddressSpace : uint32_t {
  Global = 1,
  Workgroup = 2,
  Private = 3,
};

::std::optional<AddressSpace> symbolizeAddressSpace(uint32_t);
::llvm::StringRef stringifyAddressSpace(AddressSpace);
::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAddressSpace() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(AddressSpace enumValue) {
  return stringifyAddressSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AddressSpace> symbolizeEnum<AddressSpace>(::llvm::StringRef str) {
  return symbolizeAddressSpace(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::AddressSpace, ::mlir::gpu::AddressSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::AddressSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for GPU address space");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::AddressSpace> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::AddressSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid GPU address space specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::AddressSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::AddressSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::AddressSpace getEmptyKey() {
    return static_cast<::mlir::gpu::AddressSpace>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::AddressSpace getTombstoneKey() {
    return static_cast<::mlir::gpu::AddressSpace>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::AddressSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::AddressSpace &lhs, const ::mlir::gpu::AddressSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// built-in reduction operations supported by gpu.allreduce.
enum class AllReduceOperation : uint32_t {
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
  MINIMUMF = 11,
  MAXIMUMF = 12,
};

::std::optional<AllReduceOperation> symbolizeAllReduceOperation(uint32_t);
::llvm::StringRef stringifyAllReduceOperation(AllReduceOperation);
::std::optional<AllReduceOperation> symbolizeAllReduceOperation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAllReduceOperation() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(AllReduceOperation enumValue) {
  return stringifyAllReduceOperation(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AllReduceOperation> symbolizeEnum<AllReduceOperation>(::llvm::StringRef str) {
  return symbolizeAllReduceOperation(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::AllReduceOperation, ::mlir::gpu::AllReduceOperation> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::AllReduceOperation> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for built-in reduction operations supported by gpu.allreduce.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::AllReduceOperation> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::AllReduceOperation>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid built-in reduction operations supported by gpu.allreduce. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::AllReduceOperation value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::AllReduceOperation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::AllReduceOperation getEmptyKey() {
    return static_cast<::mlir::gpu::AllReduceOperation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::AllReduceOperation getTombstoneKey() {
    return static_cast<::mlir::gpu::AllReduceOperation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::AllReduceOperation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::AllReduceOperation &lhs, const ::mlir::gpu::AllReduceOperation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// GPU compilation format
enum class CompilationTarget : uint32_t {
  Offload = 1,
  Assembly = 2,
  Binary = 3,
  Fatbin = 4,
};

::std::optional<CompilationTarget> symbolizeCompilationTarget(uint32_t);
::llvm::StringRef stringifyCompilationTarget(CompilationTarget);
::std::optional<CompilationTarget> symbolizeCompilationTarget(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCompilationTarget() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(CompilationTarget enumValue) {
  return stringifyCompilationTarget(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CompilationTarget> symbolizeEnum<CompilationTarget>(::llvm::StringRef str) {
  return symbolizeCompilationTarget(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::CompilationTarget, ::mlir::gpu::CompilationTarget> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::CompilationTarget> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for GPU compilation format");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::CompilationTarget> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::CompilationTarget>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid GPU compilation format specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::CompilationTarget value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::CompilationTarget> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::CompilationTarget getEmptyKey() {
    return static_cast<::mlir::gpu::CompilationTarget>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::CompilationTarget getTombstoneKey() {
    return static_cast<::mlir::gpu::CompilationTarget>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::CompilationTarget &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::CompilationTarget &lhs, const ::mlir::gpu::CompilationTarget &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// a dimension, either 'x', 'y', or 'z'
enum class Dimension : uint32_t {
  x = 0,
  y = 1,
  z = 2,
};

::std::optional<Dimension> symbolizeDimension(uint32_t);
::llvm::StringRef stringifyDimension(Dimension);
::std::optional<Dimension> symbolizeDimension(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDimension() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Dimension enumValue) {
  return stringifyDimension(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Dimension> symbolizeEnum<Dimension>(::llvm::StringRef str) {
  return symbolizeDimension(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Dimension, ::mlir::gpu::Dimension> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Dimension> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for a dimension, either 'x', 'y', or 'z'");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Dimension> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Dimension>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid a dimension, either 'x', 'y', or 'z' specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Dimension value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Dimension> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::Dimension getEmptyKey() {
    return static_cast<::mlir::gpu::Dimension>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Dimension getTombstoneKey() {
    return static_cast<::mlir::gpu::Dimension>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Dimension &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Dimension &lhs, const ::mlir::gpu::Dimension &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// pruning strategy for 2:4 sparse matrix
enum class Prune2To4SpMatFlag : uint32_t {
  NONE = 0,
  PRUNE_ONLY = 1,
  PRUNE_AND_CHECK = 2,
};

::std::optional<Prune2To4SpMatFlag> symbolizePrune2To4SpMatFlag(uint32_t);
::llvm::StringRef stringifyPrune2To4SpMatFlag(Prune2To4SpMatFlag);
::std::optional<Prune2To4SpMatFlag> symbolizePrune2To4SpMatFlag(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPrune2To4SpMatFlag() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Prune2To4SpMatFlag enumValue) {
  return stringifyPrune2To4SpMatFlag(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Prune2To4SpMatFlag> symbolizeEnum<Prune2To4SpMatFlag>(::llvm::StringRef str) {
  return symbolizePrune2To4SpMatFlag(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Prune2To4SpMatFlag, ::mlir::gpu::Prune2To4SpMatFlag> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Prune2To4SpMatFlag> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for pruning strategy for 2:4 sparse matrix");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Prune2To4SpMatFlag> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Prune2To4SpMatFlag>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid pruning strategy for 2:4 sparse matrix specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Prune2To4SpMatFlag value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Prune2To4SpMatFlag> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::Prune2To4SpMatFlag getEmptyKey() {
    return static_cast<::mlir::gpu::Prune2To4SpMatFlag>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Prune2To4SpMatFlag getTombstoneKey() {
    return static_cast<::mlir::gpu::Prune2To4SpMatFlag>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Prune2To4SpMatFlag &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Prune2To4SpMatFlag &lhs, const ::mlir::gpu::Prune2To4SpMatFlag &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// Indexing modes supported by gpu.shuffle.
enum class ShuffleMode : uint32_t {
  XOR = 0,
  UP = 2,
  DOWN = 1,
  IDX = 3,
};

::std::optional<ShuffleMode> symbolizeShuffleMode(uint32_t);
::llvm::StringRef stringifyShuffleMode(ShuffleMode);
::std::optional<ShuffleMode> symbolizeShuffleMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForShuffleMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ShuffleMode enumValue) {
  return stringifyShuffleMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ShuffleMode> symbolizeEnum<ShuffleMode>(::llvm::StringRef str) {
  return symbolizeShuffleMode(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::ShuffleMode, ::mlir::gpu::ShuffleMode> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::ShuffleMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Indexing modes supported by gpu.shuffle.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::ShuffleMode> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::ShuffleMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Indexing modes supported by gpu.shuffle. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::ShuffleMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::ShuffleMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::ShuffleMode getEmptyKey() {
    return static_cast<::mlir::gpu::ShuffleMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::ShuffleMode getTombstoneKey() {
    return static_cast<::mlir::gpu::ShuffleMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::ShuffleMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::ShuffleMode &lhs, const ::mlir::gpu::ShuffleMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// choose whether spgemm_work_estimation_or_compute does work estimation or compute
enum class SpGEMMWorkEstimationOrComputeKind : uint32_t {
  WORK_ESTIMATION = 0,
  COMPUTE = 1,
};

::std::optional<SpGEMMWorkEstimationOrComputeKind> symbolizeSpGEMMWorkEstimationOrComputeKind(uint32_t);
::llvm::StringRef stringifySpGEMMWorkEstimationOrComputeKind(SpGEMMWorkEstimationOrComputeKind);
::std::optional<SpGEMMWorkEstimationOrComputeKind> symbolizeSpGEMMWorkEstimationOrComputeKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSpGEMMWorkEstimationOrComputeKind() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(SpGEMMWorkEstimationOrComputeKind enumValue) {
  return stringifySpGEMMWorkEstimationOrComputeKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SpGEMMWorkEstimationOrComputeKind> symbolizeEnum<SpGEMMWorkEstimationOrComputeKind>(::llvm::StringRef str) {
  return symbolizeSpGEMMWorkEstimationOrComputeKind(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind, ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for choose whether spgemm_work_estimation_or_compute does work estimation or compute");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid choose whether spgemm_work_estimation_or_compute does work estimation or compute specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind getEmptyKey() {
    return static_cast<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind getTombstoneKey() {
    return static_cast<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind &lhs, const ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// transpose mode of sparse matrix supported by sparse tensor ops
enum class TransposeMode : uint32_t {
  NON_TRANSPOSE = 0,
  TRANSPOSE = 1,
  CONJUGATE_TRANSPOSE = 2,
};

::std::optional<TransposeMode> symbolizeTransposeMode(uint32_t);
::llvm::StringRef stringifyTransposeMode(TransposeMode);
::std::optional<TransposeMode> symbolizeTransposeMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTransposeMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(TransposeMode enumValue) {
  return stringifyTransposeMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TransposeMode> symbolizeEnum<TransposeMode>(::llvm::StringRef str) {
  return symbolizeTransposeMode(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::TransposeMode, ::mlir::gpu::TransposeMode> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::TransposeMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for transpose mode of sparse matrix supported by sparse tensor ops");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::TransposeMode> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::TransposeMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid transpose mode of sparse matrix supported by sparse tensor ops specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::TransposeMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::TransposeMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::TransposeMode getEmptyKey() {
    return static_cast<::mlir::gpu::TransposeMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::TransposeMode getTombstoneKey() {
    return static_cast<::mlir::gpu::TransposeMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::TransposeMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::TransposeMode &lhs, const ::mlir::gpu::TransposeMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// elementwise operation to apply to mma matrix
enum class MMAElementwiseOp : uint32_t {
  ADDF = 0,
  MULF = 1,
  SUBF = 2,
  MAXF = 3,
  MINF = 4,
  DIVF = 5,
  ADDI = 6,
  MULI = 7,
  SUBI = 8,
  DIVS = 9,
  DIVU = 10,
  NEGATEF = 11,
  NEGATES = 12,
  EXTF = 13,
};

::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(uint32_t);
::llvm::StringRef stringifyMMAElementwiseOp(MMAElementwiseOp);
::std::optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMMAElementwiseOp() {
  return 13;
}


inline ::llvm::StringRef stringifyEnum(MMAElementwiseOp enumValue) {
  return stringifyMMAElementwiseOp(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MMAElementwiseOp> symbolizeEnum<MMAElementwiseOp>(::llvm::StringRef str) {
  return symbolizeMMAElementwiseOp(str);
}
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::MMAElementwiseOp, ::mlir::gpu::MMAElementwiseOp> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::MMAElementwiseOp> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for elementwise operation to apply to mma matrix");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::MMAElementwiseOp> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::MMAElementwiseOp>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid elementwise operation to apply to mma matrix specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::MMAElementwiseOp value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::MMAElementwiseOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::gpu::MMAElementwiseOp getEmptyKey() {
    return static_cast<::mlir::gpu::MMAElementwiseOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::MMAElementwiseOp getTombstoneKey() {
    return static_cast<::mlir::gpu::MMAElementwiseOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::MMAElementwiseOp &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::MMAElementwiseOp &lhs, const ::mlir::gpu::MMAElementwiseOp &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// Mapping ids for loop mapping
enum class MappingId : uint64_t {
  DimX = 0,
  DimY = 1,
  DimZ = 2,
  LinearDim0 = 3,
  LinearDim1 = 4,
  LinearDim2 = 5,
  LinearDim3 = 6,
  LinearDim4 = 7,
  LinearDim5 = 8,
  LinearDim6 = 9,
  LinearDim7 = 10,
  LinearDim8 = 11,
  LinearDim9 = 12,
};

::std::optional<MappingId> symbolizeMappingId(uint64_t);
::llvm::StringRef stringifyMappingId(MappingId);
::std::optional<MappingId> symbolizeMappingId(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMappingId() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(MappingId enumValue) {
  return stringifyMappingId(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MappingId> symbolizeEnum<MappingId>(::llvm::StringRef str) {
  return symbolizeMappingId(str);
}

class MappingIdAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MappingId;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MappingIdAttr get(::mlir::MLIRContext *context, MappingId val);
  MappingId getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::MappingId, ::mlir::gpu::MappingId> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::MappingId> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Mapping ids for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::MappingId> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::MappingId>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Mapping ids for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::MappingId value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::MappingId> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::MappingId getEmptyKey() {
    return static_cast<::mlir::gpu::MappingId>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::MappingId getTombstoneKey() {
    return static_cast<::mlir::gpu::MappingId>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::MappingId &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::MappingId &lhs, const ::mlir::gpu::MappingId &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace gpu {
// processor for loop mapping
enum class Processor : uint64_t {
  BlockX = 0,
  BlockY = 1,
  BlockZ = 2,
  ThreadX = 3,
  ThreadY = 4,
  ThreadZ = 5,
  Sequential = 6,
};

::std::optional<Processor> symbolizeProcessor(uint64_t);
::llvm::StringRef stringifyProcessor(Processor);
::std::optional<Processor> symbolizeProcessor(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForProcessor() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(Processor enumValue) {
  return stringifyProcessor(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Processor> symbolizeEnum<Processor>(::llvm::StringRef str) {
  return symbolizeProcessor(str);
}

class ProcessorAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Processor;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ProcessorAttr get(::mlir::MLIRContext *context, Processor val);
  Processor getValue() const;
};
} // namespace gpu
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::gpu::Processor, ::mlir::gpu::Processor> {
  template <typename ParserT>
  static FailureOr<::mlir::gpu::Processor> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for processor for loop mapping");

    // Symbolize the keyword.
    if (::std::optional<::mlir::gpu::Processor> attr = ::mlir::gpu::symbolizeEnum<::mlir::gpu::Processor>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid processor for loop mapping specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::gpu::Processor value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::gpu::Processor> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::gpu::Processor getEmptyKey() {
    return static_cast<::mlir::gpu::Processor>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::gpu::Processor getTombstoneKey() {
    return static_cast<::mlir::gpu::Processor>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::gpu::Processor &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::gpu::Processor &lhs, const ::mlir::gpu::Processor &rhs) {
    return lhs == rhs;
  }
};
}

