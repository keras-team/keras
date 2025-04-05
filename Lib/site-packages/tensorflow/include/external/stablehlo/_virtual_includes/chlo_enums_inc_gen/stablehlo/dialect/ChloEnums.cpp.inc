/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ChloOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace chlo {
::llvm::StringRef stringifyComparisonDirection(ComparisonDirection val) {
  switch (val) {
    case ComparisonDirection::EQ: return "EQ";
    case ComparisonDirection::NE: return "NE";
    case ComparisonDirection::GE: return "GE";
    case ComparisonDirection::GT: return "GT";
    case ComparisonDirection::LE: return "LE";
    case ComparisonDirection::LT: return "LT";
  }
  return "";
}

::std::optional<ComparisonDirection> symbolizeComparisonDirection(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComparisonDirection>>(str)
      .Case("EQ", ComparisonDirection::EQ)
      .Case("NE", ComparisonDirection::NE)
      .Case("GE", ComparisonDirection::GE)
      .Case("GT", ComparisonDirection::GT)
      .Case("LE", ComparisonDirection::LE)
      .Case("LT", ComparisonDirection::LT)
      .Default(::std::nullopt);
}
::std::optional<ComparisonDirection> symbolizeComparisonDirection(uint32_t value) {
  switch (value) {
  case 0: return ComparisonDirection::EQ;
  case 1: return ComparisonDirection::NE;
  case 2: return ComparisonDirection::GE;
  case 3: return ComparisonDirection::GT;
  case 4: return ComparisonDirection::LE;
  case 5: return ComparisonDirection::LT;
  default: return ::std::nullopt;
  }
}

} // namespace chlo
} // namespace mlir

namespace mlir {
namespace chlo {
::llvm::StringRef stringifyComparisonType(ComparisonType val) {
  switch (val) {
    case ComparisonType::NOTYPE: return "NOTYPE";
    case ComparisonType::FLOAT: return "FLOAT";
    case ComparisonType::TOTALORDER: return "TOTALORDER";
    case ComparisonType::SIGNED: return "SIGNED";
    case ComparisonType::UNSIGNED: return "UNSIGNED";
  }
  return "";
}

::std::optional<ComparisonType> symbolizeComparisonType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComparisonType>>(str)
      .Case("NOTYPE", ComparisonType::NOTYPE)
      .Case("FLOAT", ComparisonType::FLOAT)
      .Case("TOTALORDER", ComparisonType::TOTALORDER)
      .Case("SIGNED", ComparisonType::SIGNED)
      .Case("UNSIGNED", ComparisonType::UNSIGNED)
      .Default(::std::nullopt);
}
::std::optional<ComparisonType> symbolizeComparisonType(uint32_t value) {
  switch (value) {
  case 0: return ComparisonType::NOTYPE;
  case 1: return ComparisonType::FLOAT;
  case 2: return ComparisonType::TOTALORDER;
  case 3: return ComparisonType::SIGNED;
  case 4: return ComparisonType::UNSIGNED;
  default: return ::std::nullopt;
  }
}

} // namespace chlo
} // namespace mlir

