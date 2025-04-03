/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: LLVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
// ATT (0) or Intel (1) asm dialect
enum class AsmDialect : uint64_t {
  AD_ATT = 0,
  AD_Intel = 1,
};

::std::optional<AsmDialect> symbolizeAsmDialect(uint64_t);
::llvm::StringRef stringifyAsmDialect(AsmDialect);
::std::optional<AsmDialect> symbolizeAsmDialect(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAsmDialect() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(AsmDialect enumValue) {
  return stringifyAsmDialect(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AsmDialect> symbolizeEnum<AsmDialect>(::llvm::StringRef str) {
  return symbolizeAsmDialect(str);
}

class AsmDialectAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AsmDialect;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AsmDialectAttr get(::mlir::MLIRContext *context, AsmDialect val);
  AsmDialect getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::AsmDialect, ::mlir::LLVM::AsmDialect> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::AsmDialect> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for ATT (0) or Intel (1) asm dialect");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::AsmDialect> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::AsmDialect>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid ATT (0) or Intel (1) asm dialect specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::AsmDialect value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AsmDialect> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AsmDialect getEmptyKey() {
    return static_cast<::mlir::LLVM::AsmDialect>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AsmDialect getTombstoneKey() {
    return static_cast<::mlir::LLVM::AsmDialect>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AsmDialect &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AsmDialect &lhs, const ::mlir::LLVM::AsmDialect &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.atomicrmw binary operations
enum class AtomicBinOp : uint64_t {
  xchg = 0,
  add = 1,
  sub = 2,
  _and = 3,
  nand = 4,
  _or = 5,
  _xor = 6,
  max = 7,
  min = 8,
  umax = 9,
  umin = 10,
  fadd = 11,
  fsub = 12,
  fmax = 13,
  fmin = 14,
  uinc_wrap = 15,
  udec_wrap = 16,
  usub_cond = 17,
  usub_sat = 18,
};

::std::optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t);
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp);
::std::optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicBinOp() {
  return 18;
}


inline ::llvm::StringRef stringifyEnum(AtomicBinOp enumValue) {
  return stringifyAtomicBinOp(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AtomicBinOp> symbolizeEnum<AtomicBinOp>(::llvm::StringRef str) {
  return symbolizeAtomicBinOp(str);
}

class AtomicBinOpAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AtomicBinOp;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AtomicBinOpAttr get(::mlir::MLIRContext *context, AtomicBinOp val);
  AtomicBinOp getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::AtomicBinOp, ::mlir::LLVM::AtomicBinOp> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::AtomicBinOp> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for llvm.atomicrmw binary operations");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::AtomicBinOp> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::AtomicBinOp>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid llvm.atomicrmw binary operations specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::AtomicBinOp value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AtomicBinOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AtomicBinOp getEmptyKey() {
    return static_cast<::mlir::LLVM::AtomicBinOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AtomicBinOp getTombstoneKey() {
    return static_cast<::mlir::LLVM::AtomicBinOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AtomicBinOp &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AtomicBinOp &lhs, const ::mlir::LLVM::AtomicBinOp &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// Atomic ordering for LLVM's memory model
enum class AtomicOrdering : uint64_t {
  not_atomic = 0,
  unordered = 1,
  monotonic = 2,
  acquire = 4,
  release = 5,
  acq_rel = 6,
  seq_cst = 7,
};

::std::optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t);
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering);
::std::optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicOrdering() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(AtomicOrdering enumValue) {
  return stringifyAtomicOrdering(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AtomicOrdering> symbolizeEnum<AtomicOrdering>(::llvm::StringRef str) {
  return symbolizeAtomicOrdering(str);
}

class AtomicOrderingAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AtomicOrdering;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AtomicOrderingAttr get(::mlir::MLIRContext *context, AtomicOrdering val);
  AtomicOrdering getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::AtomicOrdering, ::mlir::LLVM::AtomicOrdering> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::AtomicOrdering> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Atomic ordering for LLVM's memory model");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::AtomicOrdering> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::AtomicOrdering>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Atomic ordering for LLVM's memory model specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::AtomicOrdering value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AtomicOrdering> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AtomicOrdering getEmptyKey() {
    return static_cast<::mlir::LLVM::AtomicOrdering>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AtomicOrdering getTombstoneKey() {
    return static_cast<::mlir::LLVM::AtomicOrdering>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AtomicOrdering &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AtomicOrdering &lhs, const ::mlir::LLVM::AtomicOrdering &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
namespace cconv {
// Calling Conventions
enum class CConv : uint64_t {
  C = 0,
  Fast = 8,
  Cold = 9,
  GHC = 10,
  HiPE = 11,
  AnyReg = 13,
  PreserveMost = 14,
  PreserveAll = 15,
  Swift = 16,
  CXX_FAST_TLS = 17,
  Tail = 18,
  CFGuard_Check = 19,
  SwiftTail = 20,
  X86_StdCall = 64,
  X86_FastCall = 65,
  ARM_APCS = 66,
  ARM_AAPCS = 67,
  ARM_AAPCS_VFP = 68,
  MSP430_INTR = 69,
  X86_ThisCall = 70,
  PTX_Kernel = 71,
  PTX_Device = 72,
  SPIR_FUNC = 75,
  SPIR_KERNEL = 76,
  Intel_OCL_BI = 77,
  X86_64_SysV = 78,
  Win64 = 79,
  X86_VectorCall = 80,
  DUMMY_HHVM = 81,
  DUMMY_HHVM_C = 82,
  X86_INTR = 83,
  AVR_INTR = 84,
  AVR_BUILTIN = 86,
  AMDGPU_VS = 87,
  AMDGPU_GS = 88,
  AMDGPU_CS = 90,
  AMDGPU_KERNEL = 91,
  X86_RegCall = 92,
  AMDGPU_HS = 93,
  MSP430_BUILTIN = 94,
  AMDGPU_LS = 95,
  AMDGPU_ES = 96,
  AArch64_VectorCall = 97,
  AArch64_SVE_VectorCall = 98,
  WASM_EmscriptenInvoke = 99,
  AMDGPU_Gfx = 100,
  M68k_INTR = 101,
};

::std::optional<CConv> symbolizeCConv(uint64_t);
::llvm::StringRef stringifyCConv(CConv);
::std::optional<CConv> symbolizeCConv(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCConv() {
  return 101;
}


inline ::llvm::StringRef stringifyEnum(CConv enumValue) {
  return stringifyCConv(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CConv> symbolizeEnum<CConv>(::llvm::StringRef str) {
  return symbolizeCConv(str);
}

class CConvAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CConv;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CConvAttr get(::mlir::MLIRContext *context, CConv val);
  CConv getValue() const;
};
} // namespace cconv
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::cconv::CConv, ::mlir::LLVM::cconv::CConv> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::cconv::CConv> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Calling Conventions");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::cconv::CConv> attr = ::mlir::LLVM::cconv::symbolizeEnum<::mlir::LLVM::cconv::CConv>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Calling Conventions specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::cconv::CConv value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::cconv::CConv> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::cconv::CConv getEmptyKey() {
    return static_cast<::mlir::LLVM::cconv::CConv>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::cconv::CConv getTombstoneKey() {
    return static_cast<::mlir::LLVM::cconv::CConv>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::cconv::CConv &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::cconv::CConv &lhs, const ::mlir::LLVM::cconv::CConv &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
namespace comdat {
// LLVM Comdat Types
enum class Comdat : uint64_t {
  Any = 0,
  ExactMatch = 1,
  Largest = 2,
  NoDeduplicate = 3,
  SameSize = 4,
};

::std::optional<Comdat> symbolizeComdat(uint64_t);
::llvm::StringRef stringifyComdat(Comdat);
::std::optional<Comdat> symbolizeComdat(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComdat() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(Comdat enumValue) {
  return stringifyComdat(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Comdat> symbolizeEnum<Comdat>(::llvm::StringRef str) {
  return symbolizeComdat(str);
}

class ComdatAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Comdat;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ComdatAttr get(::mlir::MLIRContext *context, Comdat val);
  Comdat getValue() const;
};
} // namespace comdat
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::comdat::Comdat, ::mlir::LLVM::comdat::Comdat> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::comdat::Comdat> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM Comdat Types");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::comdat::Comdat> attr = ::mlir::LLVM::comdat::symbolizeEnum<::mlir::LLVM::comdat::Comdat>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM Comdat Types specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::comdat::Comdat value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::comdat::Comdat> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::comdat::Comdat getEmptyKey() {
    return static_cast<::mlir::LLVM::comdat::Comdat>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::comdat::Comdat getTombstoneKey() {
    return static_cast<::mlir::LLVM::comdat::Comdat>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::comdat::Comdat &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::comdat::Comdat &lhs, const ::mlir::LLVM::comdat::Comdat &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM DI flags
enum class DIFlags : uint32_t {
  Zero = 0,
  Bit0 = 1,
  Bit1 = 2,
  Private = 1,
  Protected = 2,
  Public = 3,
  FwdDecl = 4,
  AppleBlock = 8,
  ReservedBit4 = 16,
  Virtual = 32,
  Artificial = 64,
  Explicit = 128,
  Prototyped = 256,
  ObjcClassComplete = 512,
  ObjectPointer = 1024,
  Vector = 2048,
  StaticMember = 4096,
  LValueReference = 8192,
  RValueReference = 16384,
  ExportSymbols = 32768,
  SingleInheritance = 65536,
  MultipleInheritance = 65536,
  VirtualInheritance = 65536,
  IntroducedVirtual = 262144,
  BitField = 524288,
  NoReturn = 1048576,
  TypePassByValue = 4194304,
  TypePassByReference = 8388608,
  EnumClass = 16777216,
  Thunk = 33554432,
  NonTrivial = 67108864,
  BigEndian = 134217728,
  LittleEndian = 268435456,
  AllCallsDescribed = 536870912,
};

::std::optional<DIFlags> symbolizeDIFlags(uint32_t);
std::string stringifyDIFlags(DIFlags);
::std::optional<DIFlags> symbolizeDIFlags(::llvm::StringRef);

inline constexpr DIFlags operator|(DIFlags a, DIFlags b) {
  return static_cast<DIFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr DIFlags operator&(DIFlags a, DIFlags b) {
  return static_cast<DIFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr DIFlags operator^(DIFlags a, DIFlags b) {
  return static_cast<DIFlags>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr DIFlags operator~(DIFlags bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<DIFlags>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(1071513599u));
}
inline constexpr bool bitEnumContainsAll(DIFlags bits, DIFlags bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(DIFlags bits, DIFlags bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr DIFlags bitEnumClear(DIFlags bits, DIFlags bit) {
  return bits & ~bit;
}
inline constexpr DIFlags bitEnumSet(DIFlags bits, DIFlags bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(DIFlags enumValue) {
  return stringifyDIFlags(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DIFlags> symbolizeEnum<DIFlags>(::llvm::StringRef str) {
  return symbolizeDIFlags(str);
}

class DIFlagsAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DIFlags;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DIFlagsAttr get(::mlir::MLIRContext *context, DIFlags val);
  DIFlags getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::DIFlags, ::mlir::LLVM::DIFlags> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::DIFlags> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM DI flags");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::DIFlags> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::DIFlags>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM DI flags specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::DIFlags value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::LLVM::DIFlags::Public:
    return p << valueStr;
  default:
    break;
  }
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::LLVM::DIFlags>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::DIFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::LLVM::DIFlags getEmptyKey() {
    return static_cast<::mlir::LLVM::DIFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::DIFlags getTombstoneKey() {
    return static_cast<::mlir::LLVM::DIFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::DIFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::DIFlags &lhs, const ::mlir::LLVM::DIFlags &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM DISubprogram flags
enum class DISubprogramFlags : uint32_t {
  Virtual = 1,
  PureVirtual = 2,
  LocalToUnit = 4,
  Definition = 8,
  Optimized = 16,
  Pure = 32,
  Elemental = 64,
  Recursive = 128,
  MainSubprogram = 256,
  Deleted = 512,
  ObjCDirect = 2048,
};

::std::optional<DISubprogramFlags> symbolizeDISubprogramFlags(uint32_t);
std::string stringifyDISubprogramFlags(DISubprogramFlags);
::std::optional<DISubprogramFlags> symbolizeDISubprogramFlags(::llvm::StringRef);

inline constexpr DISubprogramFlags operator|(DISubprogramFlags a, DISubprogramFlags b) {
  return static_cast<DISubprogramFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr DISubprogramFlags operator&(DISubprogramFlags a, DISubprogramFlags b) {
  return static_cast<DISubprogramFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr DISubprogramFlags operator^(DISubprogramFlags a, DISubprogramFlags b) {
  return static_cast<DISubprogramFlags>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr DISubprogramFlags operator~(DISubprogramFlags bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<DISubprogramFlags>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(3071u));
}
inline constexpr bool bitEnumContainsAll(DISubprogramFlags bits, DISubprogramFlags bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(DISubprogramFlags bits, DISubprogramFlags bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr DISubprogramFlags bitEnumClear(DISubprogramFlags bits, DISubprogramFlags bit) {
  return bits & ~bit;
}
inline constexpr DISubprogramFlags bitEnumSet(DISubprogramFlags bits, DISubprogramFlags bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(DISubprogramFlags enumValue) {
  return stringifyDISubprogramFlags(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DISubprogramFlags> symbolizeEnum<DISubprogramFlags>(::llvm::StringRef str) {
  return symbolizeDISubprogramFlags(str);
}

class DISubprogramFlagsAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DISubprogramFlags;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DISubprogramFlagsAttr get(::mlir::MLIRContext *context, DISubprogramFlags val);
  DISubprogramFlags getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::DISubprogramFlags, ::mlir::LLVM::DISubprogramFlags> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::DISubprogramFlags> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM DISubprogram flags");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::DISubprogramFlags> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::DISubprogramFlags>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM DISubprogram flags specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::DISubprogramFlags value) {
  auto valueStr = stringifyEnum(value);
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::LLVM::DISubprogramFlags>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::DISubprogramFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::LLVM::DISubprogramFlags getEmptyKey() {
    return static_cast<::mlir::LLVM::DISubprogramFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::DISubprogramFlags getTombstoneKey() {
    return static_cast<::mlir::LLVM::DISubprogramFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::DISubprogramFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::DISubprogramFlags &lhs, const ::mlir::LLVM::DISubprogramFlags &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.fcmp comparison predicate
enum class FCmpPredicate : uint64_t {
  _false = 0,
  oeq = 1,
  ogt = 2,
  oge = 3,
  olt = 4,
  ole = 5,
  one = 6,
  ord = 7,
  ueq = 8,
  ugt = 9,
  uge = 10,
  ult = 11,
  ule = 12,
  une = 13,
  uno = 14,
  _true = 15,
};

::std::optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t);
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate);
::std::optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFCmpPredicate() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(FCmpPredicate enumValue) {
  return stringifyFCmpPredicate(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FCmpPredicate> symbolizeEnum<FCmpPredicate>(::llvm::StringRef str) {
  return symbolizeFCmpPredicate(str);
}

class FCmpPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FCmpPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FCmpPredicateAttr get(::mlir::MLIRContext *context, FCmpPredicate val);
  FCmpPredicate getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::FCmpPredicate, ::mlir::LLVM::FCmpPredicate> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::FCmpPredicate> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for llvm.fcmp comparison predicate");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::FCmpPredicate> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::FCmpPredicate>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid llvm.fcmp comparison predicate specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::FCmpPredicate value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::FCmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::FCmpPredicate getEmptyKey() {
    return static_cast<::mlir::LLVM::FCmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::FCmpPredicate getTombstoneKey() {
    return static_cast<::mlir::LLVM::FCmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::FCmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::FCmpPredicate &lhs, const ::mlir::LLVM::FCmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM Exception Behavior
enum class FPExceptionBehavior : uint64_t {
  Ignore = 0,
  MayTrap = 1,
  Strict = 2,
};

::std::optional<FPExceptionBehavior> symbolizeFPExceptionBehavior(uint64_t);
::llvm::StringRef stringifyFPExceptionBehavior(FPExceptionBehavior);
::std::optional<FPExceptionBehavior> symbolizeFPExceptionBehavior(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFPExceptionBehavior() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(FPExceptionBehavior enumValue) {
  return stringifyFPExceptionBehavior(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FPExceptionBehavior> symbolizeEnum<FPExceptionBehavior>(::llvm::StringRef str) {
  return symbolizeFPExceptionBehavior(str);
}

class FPExceptionBehaviorAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FPExceptionBehavior;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FPExceptionBehaviorAttr get(::mlir::MLIRContext *context, FPExceptionBehavior val);
  FPExceptionBehavior getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::FPExceptionBehavior, ::mlir::LLVM::FPExceptionBehavior> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::FPExceptionBehavior> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM Exception Behavior");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::FPExceptionBehavior> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::FPExceptionBehavior>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM Exception Behavior specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::FPExceptionBehavior value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::FPExceptionBehavior> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::FPExceptionBehavior getEmptyKey() {
    return static_cast<::mlir::LLVM::FPExceptionBehavior>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::FPExceptionBehavior getTombstoneKey() {
    return static_cast<::mlir::LLVM::FPExceptionBehavior>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::FPExceptionBehavior &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::FPExceptionBehavior &lhs, const ::mlir::LLVM::FPExceptionBehavior &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM fastmath flags
enum class FastmathFlags : uint32_t {
  none = 0,
  nnan = 1,
  ninf = 2,
  nsz = 4,
  arcp = 8,
  contract = 16,
  afn = 32,
  reassoc = 64,
  fast = 127,
};

::std::optional<FastmathFlags> symbolizeFastmathFlags(uint32_t);
std::string stringifyFastmathFlags(FastmathFlags);
::std::optional<FastmathFlags> symbolizeFastmathFlags(::llvm::StringRef);

inline constexpr FastmathFlags operator|(FastmathFlags a, FastmathFlags b) {
  return static_cast<FastmathFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr FastmathFlags operator&(FastmathFlags a, FastmathFlags b) {
  return static_cast<FastmathFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr FastmathFlags operator^(FastmathFlags a, FastmathFlags b) {
  return static_cast<FastmathFlags>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr FastmathFlags operator~(FastmathFlags bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<FastmathFlags>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(127u));
}
inline constexpr bool bitEnumContainsAll(FastmathFlags bits, FastmathFlags bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(FastmathFlags bits, FastmathFlags bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr FastmathFlags bitEnumClear(FastmathFlags bits, FastmathFlags bit) {
  return bits & ~bit;
}
inline constexpr FastmathFlags bitEnumSet(FastmathFlags bits, FastmathFlags bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(FastmathFlags enumValue) {
  return stringifyFastmathFlags(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FastmathFlags> symbolizeEnum<FastmathFlags>(::llvm::StringRef str) {
  return symbolizeFastmathFlags(str);
}
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::FastmathFlags, ::mlir::LLVM::FastmathFlags> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::FastmathFlags> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM fastmath flags");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::FastmathFlags> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::FastmathFlags>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM fastmath flags specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::FastmathFlags value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::LLVM::FastmathFlags::fast:
    return p << valueStr;
  default:
    break;
  }
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::LLVM::FastmathFlags>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::FastmathFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::LLVM::FastmathFlags getEmptyKey() {
    return static_cast<::mlir::LLVM::FastmathFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::FastmathFlags getTombstoneKey() {
    return static_cast<::mlir::LLVM::FastmathFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::FastmathFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::FastmathFlags &lhs, const ::mlir::LLVM::FastmathFlags &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
namespace framePointerKind {
// LLVM FramePointerKind
enum class FramePointerKind : uint64_t {
  None = 0,
  NonLeaf = 1,
  All = 2,
  Reserved = 3,
};

::std::optional<FramePointerKind> symbolizeFramePointerKind(uint64_t);
::llvm::StringRef stringifyFramePointerKind(FramePointerKind);
::std::optional<FramePointerKind> symbolizeFramePointerKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFramePointerKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(FramePointerKind enumValue) {
  return stringifyFramePointerKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FramePointerKind> symbolizeEnum<FramePointerKind>(::llvm::StringRef str) {
  return symbolizeFramePointerKind(str);
}

class FramePointerKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FramePointerKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FramePointerKindAttr get(::mlir::MLIRContext *context, FramePointerKind val);
  FramePointerKind getValue() const;
};
} // namespace framePointerKind
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::framePointerKind::FramePointerKind, ::mlir::LLVM::framePointerKind::FramePointerKind> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::framePointerKind::FramePointerKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM FramePointerKind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::framePointerKind::FramePointerKind> attr = ::mlir::LLVM::framePointerKind::symbolizeEnum<::mlir::LLVM::framePointerKind::FramePointerKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM FramePointerKind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::framePointerKind::FramePointerKind value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::LLVM::framePointerKind::FramePointerKind::None:
  case ::mlir::LLVM::framePointerKind::FramePointerKind::All:
  case ::mlir::LLVM::framePointerKind::FramePointerKind::Reserved:
    break;
  default:
    return p << '"' << valueStr << '"';
  }
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::framePointerKind::FramePointerKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::framePointerKind::FramePointerKind getEmptyKey() {
    return static_cast<::mlir::LLVM::framePointerKind::FramePointerKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::framePointerKind::FramePointerKind getTombstoneKey() {
    return static_cast<::mlir::LLVM::framePointerKind::FramePointerKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::framePointerKind::FramePointerKind &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::framePointerKind::FramePointerKind &lhs, const ::mlir::LLVM::framePointerKind::FramePointerKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// lvm.icmp comparison predicate
enum class ICmpPredicate : uint64_t {
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

::std::optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t);
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate);
::std::optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForICmpPredicate() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(ICmpPredicate enumValue) {
  return stringifyICmpPredicate(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ICmpPredicate> symbolizeEnum<ICmpPredicate>(::llvm::StringRef str) {
  return symbolizeICmpPredicate(str);
}

class ICmpPredicateAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ICmpPredicate;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ICmpPredicateAttr get(::mlir::MLIRContext *context, ICmpPredicate val);
  ICmpPredicate getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::ICmpPredicate, ::mlir::LLVM::ICmpPredicate> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::ICmpPredicate> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for lvm.icmp comparison predicate");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::ICmpPredicate> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::ICmpPredicate>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid lvm.icmp comparison predicate specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::ICmpPredicate value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::ICmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::ICmpPredicate getEmptyKey() {
    return static_cast<::mlir::LLVM::ICmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::ICmpPredicate getTombstoneKey() {
    return static_cast<::mlir::LLVM::ICmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::ICmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::ICmpPredicate &lhs, const ::mlir::LLVM::ICmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM integer overflow flags
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
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::IntegerOverflowFlags, ::mlir::LLVM::IntegerOverflowFlags> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::IntegerOverflowFlags> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM integer overflow flags");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::IntegerOverflowFlags> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::IntegerOverflowFlags>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM integer overflow flags specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::IntegerOverflowFlags value) {
  auto valueStr = stringifyEnum(value);
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::LLVM::IntegerOverflowFlags>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::IntegerOverflowFlags> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::LLVM::IntegerOverflowFlags getEmptyKey() {
    return static_cast<::mlir::LLVM::IntegerOverflowFlags>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::IntegerOverflowFlags getTombstoneKey() {
    return static_cast<::mlir::LLVM::IntegerOverflowFlags>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::IntegerOverflowFlags &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::IntegerOverflowFlags &lhs, const ::mlir::LLVM::IntegerOverflowFlags &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM debug emission kind
enum class DIEmissionKind : uint64_t {
  None = 0,
  Full = 1,
  LineTablesOnly = 2,
  DebugDirectivesOnly = 3,
};

::std::optional<DIEmissionKind> symbolizeDIEmissionKind(uint64_t);
::llvm::StringRef stringifyDIEmissionKind(DIEmissionKind);
::std::optional<DIEmissionKind> symbolizeDIEmissionKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDIEmissionKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(DIEmissionKind enumValue) {
  return stringifyDIEmissionKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DIEmissionKind> symbolizeEnum<DIEmissionKind>(::llvm::StringRef str) {
  return symbolizeDIEmissionKind(str);
}

class DIEmissionKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DIEmissionKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DIEmissionKindAttr get(::mlir::MLIRContext *context, DIEmissionKind val);
  DIEmissionKind getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::DIEmissionKind, ::mlir::LLVM::DIEmissionKind> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::DIEmissionKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM debug emission kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::DIEmissionKind> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::DIEmissionKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM debug emission kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::DIEmissionKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::DIEmissionKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::DIEmissionKind getEmptyKey() {
    return static_cast<::mlir::LLVM::DIEmissionKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::DIEmissionKind getTombstoneKey() {
    return static_cast<::mlir::LLVM::DIEmissionKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::DIEmissionKind &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::DIEmissionKind &lhs, const ::mlir::LLVM::DIEmissionKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM debug name table kind
enum class DINameTableKind : uint64_t {
  Default = 0,
  GNU = 1,
  None = 2,
  Apple = 3,
};

::std::optional<DINameTableKind> symbolizeDINameTableKind(uint64_t);
::llvm::StringRef stringifyDINameTableKind(DINameTableKind);
::std::optional<DINameTableKind> symbolizeDINameTableKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDINameTableKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(DINameTableKind enumValue) {
  return stringifyDINameTableKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DINameTableKind> symbolizeEnum<DINameTableKind>(::llvm::StringRef str) {
  return symbolizeDINameTableKind(str);
}

class DINameTableKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DINameTableKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DINameTableKindAttr get(::mlir::MLIRContext *context, DINameTableKind val);
  DINameTableKind getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::DINameTableKind, ::mlir::LLVM::DINameTableKind> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::DINameTableKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM debug name table kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::DINameTableKind> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::DINameTableKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM debug name table kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::DINameTableKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::DINameTableKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::DINameTableKind getEmptyKey() {
    return static_cast<::mlir::LLVM::DINameTableKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::DINameTableKind getTombstoneKey() {
    return static_cast<::mlir::LLVM::DINameTableKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::DINameTableKind &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::DINameTableKind &lhs, const ::mlir::LLVM::DINameTableKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
namespace linkage {
// LLVM linkage types
enum class Linkage : uint64_t {
  Private = 0,
  Internal = 1,
  AvailableExternally = 2,
  Linkonce = 3,
  Weak = 4,
  Common = 5,
  Appending = 6,
  ExternWeak = 7,
  LinkonceODR = 8,
  WeakODR = 9,
  External = 10,
};

::std::optional<Linkage> symbolizeLinkage(uint64_t);
::llvm::StringRef stringifyLinkage(Linkage);
::std::optional<Linkage> symbolizeLinkage(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLinkage() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(Linkage enumValue) {
  return stringifyLinkage(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Linkage> symbolizeEnum<Linkage>(::llvm::StringRef str) {
  return symbolizeLinkage(str);
}

class LinkageAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Linkage;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LinkageAttr get(::mlir::MLIRContext *context, Linkage val);
  Linkage getValue() const;
};
} // namespace linkage
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::linkage::Linkage, ::mlir::LLVM::linkage::Linkage> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::linkage::Linkage> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM linkage types");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::linkage::Linkage> attr = ::mlir::LLVM::linkage::symbolizeEnum<::mlir::LLVM::linkage::Linkage>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM linkage types specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::linkage::Linkage value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::linkage::Linkage> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::linkage::Linkage getEmptyKey() {
    return static_cast<::mlir::LLVM::linkage::Linkage>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::linkage::Linkage getTombstoneKey() {
    return static_cast<::mlir::LLVM::linkage::Linkage>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::linkage::Linkage &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::linkage::Linkage &lhs, const ::mlir::LLVM::linkage::Linkage &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM ModRefInfo
enum class ModRefInfo : uint64_t {
  NoModRef = 0,
  Ref = 1,
  Mod = 2,
  ModRef = 3,
};

::std::optional<ModRefInfo> symbolizeModRefInfo(uint64_t);
::llvm::StringRef stringifyModRefInfo(ModRefInfo);
::std::optional<ModRefInfo> symbolizeModRefInfo(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForModRefInfo() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ModRefInfo enumValue) {
  return stringifyModRefInfo(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ModRefInfo> symbolizeEnum<ModRefInfo>(::llvm::StringRef str) {
  return symbolizeModRefInfo(str);
}

class ModRefInfoAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ModRefInfo;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ModRefInfoAttr get(::mlir::MLIRContext *context, ModRefInfo val);
  ModRefInfo getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::ModRefInfo, ::mlir::LLVM::ModRefInfo> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::ModRefInfo> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM ModRefInfo");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::ModRefInfo> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::ModRefInfo>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM ModRefInfo specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::ModRefInfo value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::ModRefInfo> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::ModRefInfo getEmptyKey() {
    return static_cast<::mlir::LLVM::ModRefInfo>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::ModRefInfo getTombstoneKey() {
    return static_cast<::mlir::LLVM::ModRefInfo>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::ModRefInfo &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::ModRefInfo &lhs, const ::mlir::LLVM::ModRefInfo &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM Rounding Mode
enum class RoundingMode : uint64_t {
  TowardZero = 0,
  NearestTiesToEven = 1,
  TowardPositive = 2,
  TowardNegative = 3,
  NearestTiesToAway = 4,
  Dynamic = 7,
  Invalid,
};

::llvm::StringRef stringifyRoundingMode(RoundingMode);
::std::optional<RoundingMode> symbolizeRoundingMode(::llvm::StringRef);

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
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::RoundingMode, ::mlir::LLVM::RoundingMode> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::RoundingMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM Rounding Mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::RoundingMode> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::RoundingMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM Rounding Mode specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::RoundingMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::RoundingMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::RoundingMode getEmptyKey() {
    return static_cast<::mlir::LLVM::RoundingMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::RoundingMode getTombstoneKey() {
    return static_cast<::mlir::LLVM::RoundingMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::RoundingMode &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::RoundingMode &lhs, const ::mlir::LLVM::RoundingMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
namespace tailcallkind {
// Tail Call Kind
enum class TailCallKind : uint64_t {
  None = 0,
  NoTail = 3,
  MustTail = 2,
  Tail = 1,
};

::std::optional<TailCallKind> symbolizeTailCallKind(uint64_t);
::llvm::StringRef stringifyTailCallKind(TailCallKind);
::std::optional<TailCallKind> symbolizeTailCallKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTailCallKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(TailCallKind enumValue) {
  return stringifyTailCallKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TailCallKind> symbolizeEnum<TailCallKind>(::llvm::StringRef str) {
  return symbolizeTailCallKind(str);
}

class TailCallKindAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = TailCallKind;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static TailCallKindAttr get(::mlir::MLIRContext *context, TailCallKind val);
  TailCallKind getValue() const;
};
} // namespace tailcallkind
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::tailcallkind::TailCallKind, ::mlir::LLVM::tailcallkind::TailCallKind> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::tailcallkind::TailCallKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Tail Call Kind");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::tailcallkind::TailCallKind> attr = ::mlir::LLVM::tailcallkind::symbolizeEnum<::mlir::LLVM::tailcallkind::TailCallKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Tail Call Kind specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::tailcallkind::TailCallKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::tailcallkind::TailCallKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::tailcallkind::TailCallKind getEmptyKey() {
    return static_cast<::mlir::LLVM::tailcallkind::TailCallKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::tailcallkind::TailCallKind getTombstoneKey() {
    return static_cast<::mlir::LLVM::tailcallkind::TailCallKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::tailcallkind::TailCallKind &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::tailcallkind::TailCallKind &lhs, const ::mlir::LLVM::tailcallkind::TailCallKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM GlobalValue UnnamedAddr
enum class UnnamedAddr : uint64_t {
  None = 0,
  Local = 1,
  Global = 2,
};

::std::optional<UnnamedAddr> symbolizeUnnamedAddr(uint64_t);
::llvm::StringRef stringifyUnnamedAddr(UnnamedAddr);
::std::optional<UnnamedAddr> symbolizeUnnamedAddr(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForUnnamedAddr() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(UnnamedAddr enumValue) {
  return stringifyUnnamedAddr(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<UnnamedAddr> symbolizeEnum<UnnamedAddr>(::llvm::StringRef str) {
  return symbolizeUnnamedAddr(str);
}

class UnnamedAddrAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = UnnamedAddr;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static UnnamedAddrAttr get(::mlir::MLIRContext *context, UnnamedAddr val);
  UnnamedAddr getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::UnnamedAddr, ::mlir::LLVM::UnnamedAddr> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::UnnamedAddr> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM GlobalValue UnnamedAddr");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::UnnamedAddr> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::UnnamedAddr>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM GlobalValue UnnamedAddr specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::UnnamedAddr value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::LLVM::UnnamedAddr::Local:
  case ::mlir::LLVM::UnnamedAddr::Global:
    break;
  default:
    return p << '"' << valueStr << '"';
  }
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::UnnamedAddr> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::UnnamedAddr getEmptyKey() {
    return static_cast<::mlir::LLVM::UnnamedAddr>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::UnnamedAddr getTombstoneKey() {
    return static_cast<::mlir::LLVM::UnnamedAddr>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::UnnamedAddr &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::UnnamedAddr &lhs, const ::mlir::LLVM::UnnamedAddr &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM GlobalValue Visibility
enum class Visibility : uint64_t {
  Default = 0,
  Hidden = 1,
  Protected = 2,
};

::std::optional<Visibility> symbolizeVisibility(uint64_t);
::llvm::StringRef stringifyVisibility(Visibility);
::std::optional<Visibility> symbolizeVisibility(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForVisibility() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Visibility enumValue) {
  return stringifyVisibility(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Visibility> symbolizeEnum<Visibility>(::llvm::StringRef str) {
  return symbolizeVisibility(str);
}

class VisibilityAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Visibility;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static VisibilityAttr get(::mlir::MLIRContext *context, Visibility val);
  Visibility getValue() const;
};
} // namespace LLVM
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::LLVM::Visibility, ::mlir::LLVM::Visibility> {
  template <typename ParserT>
  static FailureOr<::mlir::LLVM::Visibility> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for LLVM GlobalValue Visibility");

    // Symbolize the keyword.
    if (::std::optional<::mlir::LLVM::Visibility> attr = ::mlir::LLVM::symbolizeEnum<::mlir::LLVM::Visibility>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid LLVM GlobalValue Visibility specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::LLVM::Visibility value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::LLVM::Visibility::Hidden:
  case ::mlir::LLVM::Visibility::Protected:
    break;
  default:
    return p << '"' << valueStr << '"';
  }
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::Visibility> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::Visibility getEmptyKey() {
    return static_cast<::mlir::LLVM::Visibility>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::Visibility getTombstoneKey() {
    return static_cast<::mlir::LLVM::Visibility>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::Visibility &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::Visibility &lhs, const ::mlir::LLVM::Visibility &rhs) {
    return lhs == rhs;
  }
};
}

