/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: VectorTransformsBase.td                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
// control the lowering of `vector.contract` operations.
enum class VectorContractLowering : uint32_t {
  Dot = 0,
  Matmul = 1,
  OuterProduct = 2,
  ParallelArith = 3,
};

::std::optional<VectorContractLowering> symbolizeVectorContractLowering(uint32_t);
::llvm::StringRef stringifyVectorContractLowering(VectorContractLowering);
::std::optional<VectorContractLowering> symbolizeVectorContractLowering(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVectorContractLowering() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(VectorContractLowering enumValue) {
  return stringifyVectorContractLowering(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<VectorContractLowering> symbolizeEnum<VectorContractLowering>(::llvm::StringRef str) {
  return symbolizeVectorContractLowering(str);
}

class VectorContractLoweringAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = VectorContractLowering;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static VectorContractLoweringAttr get(::mlir::MLIRContext *context, VectorContractLowering val);
  VectorContractLowering getValue() const;
};
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::VectorContractLowering, ::mlir::vector::VectorContractLowering> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::VectorContractLowering> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for control the lowering of `vector.contract` operations.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::VectorContractLowering> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::VectorContractLowering>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid control the lowering of `vector.contract` operations. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::VectorContractLowering value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::VectorContractLowering> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::VectorContractLowering getEmptyKey() {
    return static_cast<::mlir::vector::VectorContractLowering>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::VectorContractLowering getTombstoneKey() {
    return static_cast<::mlir::vector::VectorContractLowering>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::VectorContractLowering &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::VectorContractLowering &lhs, const ::mlir::vector::VectorContractLowering &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vector {
// control the lowering of `vector.multi_reduction`.
enum class VectorMultiReductionLowering : uint32_t {
  InnerParallel = 0,
  InnerReduction = 1,
};

::std::optional<VectorMultiReductionLowering> symbolizeVectorMultiReductionLowering(uint32_t);
::llvm::StringRef stringifyVectorMultiReductionLowering(VectorMultiReductionLowering);
::std::optional<VectorMultiReductionLowering> symbolizeVectorMultiReductionLowering(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVectorMultiReductionLowering() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(VectorMultiReductionLowering enumValue) {
  return stringifyVectorMultiReductionLowering(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<VectorMultiReductionLowering> symbolizeEnum<VectorMultiReductionLowering>(::llvm::StringRef str) {
  return symbolizeVectorMultiReductionLowering(str);
}

class VectorMultiReductionLoweringAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = VectorMultiReductionLowering;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static VectorMultiReductionLoweringAttr get(::mlir::MLIRContext *context, VectorMultiReductionLowering val);
  VectorMultiReductionLowering getValue() const;
};
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::VectorMultiReductionLowering, ::mlir::vector::VectorMultiReductionLowering> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::VectorMultiReductionLowering> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for control the lowering of `vector.multi_reduction`.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::VectorMultiReductionLowering> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::VectorMultiReductionLowering>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid control the lowering of `vector.multi_reduction`. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::VectorMultiReductionLowering value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::VectorMultiReductionLowering> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::VectorMultiReductionLowering getEmptyKey() {
    return static_cast<::mlir::vector::VectorMultiReductionLowering>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::VectorMultiReductionLowering getTombstoneKey() {
    return static_cast<::mlir::vector::VectorMultiReductionLowering>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::VectorMultiReductionLowering &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::VectorMultiReductionLowering &lhs, const ::mlir::vector::VectorMultiReductionLowering &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vector {
// control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants.
enum class VectorTransferSplit : uint32_t {
  None = 0,
  VectorTransfer = 1,
  LinalgCopy = 2,
  ForceInBounds = 3,
};

::std::optional<VectorTransferSplit> symbolizeVectorTransferSplit(uint32_t);
::llvm::StringRef stringifyVectorTransferSplit(VectorTransferSplit);
::std::optional<VectorTransferSplit> symbolizeVectorTransferSplit(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVectorTransferSplit() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(VectorTransferSplit enumValue) {
  return stringifyVectorTransferSplit(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<VectorTransferSplit> symbolizeEnum<VectorTransferSplit>(::llvm::StringRef str) {
  return symbolizeVectorTransferSplit(str);
}

class VectorTransferSplitAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = VectorTransferSplit;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static VectorTransferSplitAttr get(::mlir::MLIRContext *context, VectorTransferSplit val);
  VectorTransferSplit getValue() const;
};
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::VectorTransferSplit, ::mlir::vector::VectorTransferSplit> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::VectorTransferSplit> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::VectorTransferSplit> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::VectorTransferSplit>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::VectorTransferSplit value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::vector::VectorTransferSplit::None:
    break;
  default:
    return p << '"' << valueStr << '"';
  }
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::VectorTransferSplit> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::VectorTransferSplit getEmptyKey() {
    return static_cast<::mlir::vector::VectorTransferSplit>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::VectorTransferSplit getTombstoneKey() {
    return static_cast<::mlir::vector::VectorTransferSplit>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::VectorTransferSplit &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::VectorTransferSplit &lhs, const ::mlir::vector::VectorTransferSplit &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace vector {
// control the lowering of `vector.transpose` operations.
enum class VectorTransposeLowering : uint32_t {
  EltWise = 0,
  Flat = 1,
  Shuffle1D = 2,
  Shuffle16x16 = 3,
};

::std::optional<VectorTransposeLowering> symbolizeVectorTransposeLowering(uint32_t);
::llvm::StringRef stringifyVectorTransposeLowering(VectorTransposeLowering);
::std::optional<VectorTransposeLowering> symbolizeVectorTransposeLowering(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVectorTransposeLowering() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(VectorTransposeLowering enumValue) {
  return stringifyVectorTransposeLowering(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<VectorTransposeLowering> symbolizeEnum<VectorTransposeLowering>(::llvm::StringRef str) {
  return symbolizeVectorTransposeLowering(str);
}

class VectorTransposeLoweringAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = VectorTransposeLowering;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static VectorTransposeLoweringAttr get(::mlir::MLIRContext *context, VectorTransposeLowering val);
  VectorTransposeLowering getValue() const;
};
} // namespace vector
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::vector::VectorTransposeLowering, ::mlir::vector::VectorTransposeLowering> {
  template <typename ParserT>
  static FailureOr<::mlir::vector::VectorTransposeLowering> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for control the lowering of `vector.transpose` operations.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::vector::VectorTransposeLowering> attr = ::mlir::vector::symbolizeEnum<::mlir::vector::VectorTransposeLowering>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid control the lowering of `vector.transpose` operations. specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::vector::VectorTransposeLowering value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::vector::VectorTransposeLowering> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::vector::VectorTransposeLowering getEmptyKey() {
    return static_cast<::mlir::vector::VectorTransposeLowering>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::vector::VectorTransposeLowering getTombstoneKey() {
    return static_cast<::mlir::vector::VectorTransposeLowering>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::vector::VectorTransposeLowering &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::vector::VectorTransposeLowering &lhs, const ::mlir::vector::VectorTransposeLowering &rhs) {
    return lhs == rhs;
  }
};
}

