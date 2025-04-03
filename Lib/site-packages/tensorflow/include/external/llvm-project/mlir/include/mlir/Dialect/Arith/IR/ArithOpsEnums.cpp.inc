/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArithBase.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace arith {
::llvm::StringRef stringifyCmpFPredicate(CmpFPredicate val) {
  switch (val) {
    case CmpFPredicate::AlwaysFalse: return "false";
    case CmpFPredicate::OEQ: return "oeq";
    case CmpFPredicate::OGT: return "ogt";
    case CmpFPredicate::OGE: return "oge";
    case CmpFPredicate::OLT: return "olt";
    case CmpFPredicate::OLE: return "ole";
    case CmpFPredicate::ONE: return "one";
    case CmpFPredicate::ORD: return "ord";
    case CmpFPredicate::UEQ: return "ueq";
    case CmpFPredicate::UGT: return "ugt";
    case CmpFPredicate::UGE: return "uge";
    case CmpFPredicate::ULT: return "ult";
    case CmpFPredicate::ULE: return "ule";
    case CmpFPredicate::UNE: return "une";
    case CmpFPredicate::UNO: return "uno";
    case CmpFPredicate::AlwaysTrue: return "true";
  }
  return "";
}

::std::optional<CmpFPredicate> symbolizeCmpFPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CmpFPredicate>>(str)
      .Case("false", CmpFPredicate::AlwaysFalse)
      .Case("oeq", CmpFPredicate::OEQ)
      .Case("ogt", CmpFPredicate::OGT)
      .Case("oge", CmpFPredicate::OGE)
      .Case("olt", CmpFPredicate::OLT)
      .Case("ole", CmpFPredicate::OLE)
      .Case("one", CmpFPredicate::ONE)
      .Case("ord", CmpFPredicate::ORD)
      .Case("ueq", CmpFPredicate::UEQ)
      .Case("ugt", CmpFPredicate::UGT)
      .Case("uge", CmpFPredicate::UGE)
      .Case("ult", CmpFPredicate::ULT)
      .Case("ule", CmpFPredicate::ULE)
      .Case("une", CmpFPredicate::UNE)
      .Case("uno", CmpFPredicate::UNO)
      .Case("true", CmpFPredicate::AlwaysTrue)
      .Default(::std::nullopt);
}
::std::optional<CmpFPredicate> symbolizeCmpFPredicate(uint64_t value) {
  switch (value) {
  case 0: return CmpFPredicate::AlwaysFalse;
  case 1: return CmpFPredicate::OEQ;
  case 2: return CmpFPredicate::OGT;
  case 3: return CmpFPredicate::OGE;
  case 4: return CmpFPredicate::OLT;
  case 5: return CmpFPredicate::OLE;
  case 6: return CmpFPredicate::ONE;
  case 7: return CmpFPredicate::ORD;
  case 8: return CmpFPredicate::UEQ;
  case 9: return CmpFPredicate::UGT;
  case 10: return CmpFPredicate::UGE;
  case 11: return CmpFPredicate::ULT;
  case 12: return CmpFPredicate::ULE;
  case 13: return CmpFPredicate::UNE;
  case 14: return CmpFPredicate::UNO;
  case 15: return CmpFPredicate::AlwaysTrue;
  default: return ::std::nullopt;
  }
}

bool CmpFPredicateAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 11)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 12)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 13)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 14)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 15)));
}
CmpFPredicateAttr CmpFPredicateAttr::get(::mlir::MLIRContext *context, CmpFPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<CmpFPredicateAttr>(baseAttr);
}
CmpFPredicate CmpFPredicateAttr::getValue() const {
  return static_cast<CmpFPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace arith
} // namespace mlir

namespace mlir {
namespace arith {
::llvm::StringRef stringifyCmpIPredicate(CmpIPredicate val) {
  switch (val) {
    case CmpIPredicate::eq: return "eq";
    case CmpIPredicate::ne: return "ne";
    case CmpIPredicate::slt: return "slt";
    case CmpIPredicate::sle: return "sle";
    case CmpIPredicate::sgt: return "sgt";
    case CmpIPredicate::sge: return "sge";
    case CmpIPredicate::ult: return "ult";
    case CmpIPredicate::ule: return "ule";
    case CmpIPredicate::ugt: return "ugt";
    case CmpIPredicate::uge: return "uge";
  }
  return "";
}

::std::optional<CmpIPredicate> symbolizeCmpIPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CmpIPredicate>>(str)
      .Case("eq", CmpIPredicate::eq)
      .Case("ne", CmpIPredicate::ne)
      .Case("slt", CmpIPredicate::slt)
      .Case("sle", CmpIPredicate::sle)
      .Case("sgt", CmpIPredicate::sgt)
      .Case("sge", CmpIPredicate::sge)
      .Case("ult", CmpIPredicate::ult)
      .Case("ule", CmpIPredicate::ule)
      .Case("ugt", CmpIPredicate::ugt)
      .Case("uge", CmpIPredicate::uge)
      .Default(::std::nullopt);
}
::std::optional<CmpIPredicate> symbolizeCmpIPredicate(uint64_t value) {
  switch (value) {
  case 0: return CmpIPredicate::eq;
  case 1: return CmpIPredicate::ne;
  case 2: return CmpIPredicate::slt;
  case 3: return CmpIPredicate::sle;
  case 4: return CmpIPredicate::sgt;
  case 5: return CmpIPredicate::sge;
  case 6: return CmpIPredicate::ult;
  case 7: return CmpIPredicate::ule;
  case 8: return CmpIPredicate::ugt;
  case 9: return CmpIPredicate::uge;
  default: return ::std::nullopt;
  }
}

bool CmpIPredicateAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)));
}
CmpIPredicateAttr CmpIPredicateAttr::get(::mlir::MLIRContext *context, CmpIPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<CmpIPredicateAttr>(baseAttr);
}
CmpIPredicate CmpIPredicateAttr::getValue() const {
  return static_cast<CmpIPredicate>(::mlir::IntegerAttr::getInt());
}
} // namespace arith
} // namespace mlir

namespace mlir {
namespace arith {
std::string stringifyIntegerOverflowFlags(IntegerOverflowFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(3u == (3u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "none";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (1u == (1u & val))
    strs.push_back("nsw");

  if (2u == (2u & val))
    strs.push_back("nuw");
  return ::llvm::join(strs, ", ");
}

::std::optional<IntegerOverflowFlags> symbolizeIntegerOverflowFlags(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "none") return IntegerOverflowFlags::none;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("nsw", 1)
      .Case("nuw", 2)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<IntegerOverflowFlags>(val);
}

::std::optional<IntegerOverflowFlags> symbolizeIntegerOverflowFlags(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return IntegerOverflowFlags::none;

  if (value & ~static_cast<uint32_t>(3u)) return std::nullopt;
  return static_cast<IntegerOverflowFlags>(value);
}
} // namespace arith
} // namespace mlir

namespace mlir {
namespace arith {
::llvm::StringRef stringifyRoundingMode(RoundingMode val) {
  switch (val) {
    case RoundingMode::to_nearest_even: return "to_nearest_even";
    case RoundingMode::downward: return "downward";
    case RoundingMode::upward: return "upward";
    case RoundingMode::toward_zero: return "toward_zero";
    case RoundingMode::to_nearest_away: return "to_nearest_away";
  }
  return "";
}

::std::optional<RoundingMode> symbolizeRoundingMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RoundingMode>>(str)
      .Case("to_nearest_even", RoundingMode::to_nearest_even)
      .Case("downward", RoundingMode::downward)
      .Case("upward", RoundingMode::upward)
      .Case("toward_zero", RoundingMode::toward_zero)
      .Case("to_nearest_away", RoundingMode::to_nearest_away)
      .Default(::std::nullopt);
}
::std::optional<RoundingMode> symbolizeRoundingMode(uint32_t value) {
  switch (value) {
  case 0: return RoundingMode::to_nearest_even;
  case 1: return RoundingMode::downward;
  case 2: return RoundingMode::upward;
  case 3: return RoundingMode::toward_zero;
  case 4: return RoundingMode::to_nearest_away;
  default: return ::std::nullopt;
  }
}

bool RoundingModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)));
}
RoundingModeAttr RoundingModeAttr::get(::mlir::MLIRContext *context, RoundingMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<RoundingModeAttr>(baseAttr);
}
RoundingMode RoundingModeAttr::getValue() const {
  return static_cast<RoundingMode>(::mlir::IntegerAttr::getInt());
}
} // namespace arith
} // namespace mlir

namespace mlir {
namespace arith {
::llvm::StringRef stringifyAtomicRMWKind(AtomicRMWKind val) {
  switch (val) {
    case AtomicRMWKind::addf: return "addf";
    case AtomicRMWKind::addi: return "addi";
    case AtomicRMWKind::assign: return "assign";
    case AtomicRMWKind::maximumf: return "maximumf";
    case AtomicRMWKind::maxs: return "maxs";
    case AtomicRMWKind::maxu: return "maxu";
    case AtomicRMWKind::minimumf: return "minimumf";
    case AtomicRMWKind::mins: return "mins";
    case AtomicRMWKind::minu: return "minu";
    case AtomicRMWKind::mulf: return "mulf";
    case AtomicRMWKind::muli: return "muli";
    case AtomicRMWKind::ori: return "ori";
    case AtomicRMWKind::andi: return "andi";
    case AtomicRMWKind::maxnumf: return "maxnumf";
    case AtomicRMWKind::minnumf: return "minnumf";
  }
  return "";
}

::std::optional<AtomicRMWKind> symbolizeAtomicRMWKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AtomicRMWKind>>(str)
      .Case("addf", AtomicRMWKind::addf)
      .Case("addi", AtomicRMWKind::addi)
      .Case("assign", AtomicRMWKind::assign)
      .Case("maximumf", AtomicRMWKind::maximumf)
      .Case("maxs", AtomicRMWKind::maxs)
      .Case("maxu", AtomicRMWKind::maxu)
      .Case("minimumf", AtomicRMWKind::minimumf)
      .Case("mins", AtomicRMWKind::mins)
      .Case("minu", AtomicRMWKind::minu)
      .Case("mulf", AtomicRMWKind::mulf)
      .Case("muli", AtomicRMWKind::muli)
      .Case("ori", AtomicRMWKind::ori)
      .Case("andi", AtomicRMWKind::andi)
      .Case("maxnumf", AtomicRMWKind::maxnumf)
      .Case("minnumf", AtomicRMWKind::minnumf)
      .Default(::std::nullopt);
}
::std::optional<AtomicRMWKind> symbolizeAtomicRMWKind(uint64_t value) {
  switch (value) {
  case 0: return AtomicRMWKind::addf;
  case 1: return AtomicRMWKind::addi;
  case 2: return AtomicRMWKind::assign;
  case 3: return AtomicRMWKind::maximumf;
  case 4: return AtomicRMWKind::maxs;
  case 5: return AtomicRMWKind::maxu;
  case 6: return AtomicRMWKind::minimumf;
  case 7: return AtomicRMWKind::mins;
  case 8: return AtomicRMWKind::minu;
  case 9: return AtomicRMWKind::mulf;
  case 10: return AtomicRMWKind::muli;
  case 11: return AtomicRMWKind::ori;
  case 12: return AtomicRMWKind::andi;
  case 13: return AtomicRMWKind::maxnumf;
  case 14: return AtomicRMWKind::minnumf;
  default: return ::std::nullopt;
  }
}

bool AtomicRMWKindAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 3)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 4)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 5)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 6)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 7)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 8)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 9)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 10)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 11)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 12)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 13)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 14)));
}
AtomicRMWKindAttr AtomicRMWKindAttr::get(::mlir::MLIRContext *context, AtomicRMWKind val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<AtomicRMWKindAttr>(baseAttr);
}
AtomicRMWKind AtomicRMWKindAttr::getValue() const {
  return static_cast<AtomicRMWKind>(::mlir::IntegerAttr::getInt());
}
} // namespace arith
} // namespace mlir

namespace mlir {
namespace arith {
std::string stringifyFastMathFlags(FastMathFlags symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(127u == (127u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "none";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (127u == (127u & val)) {
    strs.push_back("fast");
    val &= ~static_cast<uint32_t>(127);
  }

  if (1u == (1u & val))
    strs.push_back("reassoc");

  if (2u == (2u & val))
    strs.push_back("nnan");

  if (4u == (4u & val))
    strs.push_back("ninf");

  if (8u == (8u & val))
    strs.push_back("nsz");

  if (16u == (16u & val))
    strs.push_back("arcp");

  if (32u == (32u & val))
    strs.push_back("contract");

  if (64u == (64u & val))
    strs.push_back("afn");
  return ::llvm::join(strs, ",");
}

::std::optional<FastMathFlags> symbolizeFastMathFlags(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "none") return FastMathFlags::none;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("reassoc", 1)
      .Case("nnan", 2)
      .Case("ninf", 4)
      .Case("nsz", 8)
      .Case("arcp", 16)
      .Case("contract", 32)
      .Case("afn", 64)
      .Case("fast", 127)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<FastMathFlags>(val);
}

::std::optional<FastMathFlags> symbolizeFastMathFlags(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return FastMathFlags::none;

  if (value & ~static_cast<uint32_t>(127u)) return std::nullopt;
  return static_cast<FastMathFlags>(value);
}
} // namespace arith
} // namespace mlir

