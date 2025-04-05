/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: VectorAttributes.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
::llvm::StringRef stringifyCombiningKind(CombiningKind val) {
  switch (val) {
    case CombiningKind::ADD: return "add";
    case CombiningKind::MUL: return "mul";
    case CombiningKind::MINUI: return "minui";
    case CombiningKind::MINSI: return "minsi";
    case CombiningKind::MINNUMF: return "minnumf";
    case CombiningKind::MAXUI: return "maxui";
    case CombiningKind::MAXSI: return "maxsi";
    case CombiningKind::MAXNUMF: return "maxnumf";
    case CombiningKind::AND: return "and";
    case CombiningKind::OR: return "or";
    case CombiningKind::XOR: return "xor";
    case CombiningKind::MAXIMUMF: return "maximumf";
    case CombiningKind::MINIMUMF: return "minimumf";
  }
  return "";
}

::std::optional<CombiningKind> symbolizeCombiningKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CombiningKind>>(str)
      .Case("add", CombiningKind::ADD)
      .Case("mul", CombiningKind::MUL)
      .Case("minui", CombiningKind::MINUI)
      .Case("minsi", CombiningKind::MINSI)
      .Case("minnumf", CombiningKind::MINNUMF)
      .Case("maxui", CombiningKind::MAXUI)
      .Case("maxsi", CombiningKind::MAXSI)
      .Case("maxnumf", CombiningKind::MAXNUMF)
      .Case("and", CombiningKind::AND)
      .Case("or", CombiningKind::OR)
      .Case("xor", CombiningKind::XOR)
      .Case("maximumf", CombiningKind::MAXIMUMF)
      .Case("minimumf", CombiningKind::MINIMUMF)
      .Default(::std::nullopt);
}
::std::optional<CombiningKind> symbolizeCombiningKind(uint32_t value) {
  switch (value) {
  case 0: return CombiningKind::ADD;
  case 1: return CombiningKind::MUL;
  case 2: return CombiningKind::MINUI;
  case 3: return CombiningKind::MINSI;
  case 4: return CombiningKind::MINNUMF;
  case 5: return CombiningKind::MAXUI;
  case 6: return CombiningKind::MAXSI;
  case 7: return CombiningKind::MAXNUMF;
  case 8: return CombiningKind::AND;
  case 9: return CombiningKind::OR;
  case 10: return CombiningKind::XOR;
  case 12: return CombiningKind::MAXIMUMF;
  case 11: return CombiningKind::MINIMUMF;
  default: return ::std::nullopt;
  }
}

} // namespace vector
} // namespace mlir

namespace mlir {
namespace vector {
::llvm::StringRef stringifyPrintPunctuation(PrintPunctuation val) {
  switch (val) {
    case PrintPunctuation::NoPunctuation: return "no_punctuation";
    case PrintPunctuation::NewLine: return "newline";
    case PrintPunctuation::Comma: return "comma";
    case PrintPunctuation::Open: return "open";
    case PrintPunctuation::Close: return "close";
  }
  return "";
}

::std::optional<PrintPunctuation> symbolizePrintPunctuation(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<PrintPunctuation>>(str)
      .Case("no_punctuation", PrintPunctuation::NoPunctuation)
      .Case("newline", PrintPunctuation::NewLine)
      .Case("comma", PrintPunctuation::Comma)
      .Case("open", PrintPunctuation::Open)
      .Case("close", PrintPunctuation::Close)
      .Default(::std::nullopt);
}
::std::optional<PrintPunctuation> symbolizePrintPunctuation(uint32_t value) {
  switch (value) {
  case 0: return PrintPunctuation::NoPunctuation;
  case 1: return PrintPunctuation::NewLine;
  case 2: return PrintPunctuation::Comma;
  case 3: return PrintPunctuation::Open;
  case 4: return PrintPunctuation::Close;
  default: return ::std::nullopt;
  }
}

} // namespace vector
} // namespace mlir

namespace mlir {
namespace vector {
::llvm::StringRef stringifyIteratorType(IteratorType val) {
  switch (val) {
    case IteratorType::parallel: return "parallel";
    case IteratorType::reduction: return "reduction";
  }
  return "";
}

::std::optional<IteratorType> symbolizeIteratorType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<IteratorType>>(str)
      .Case("parallel", IteratorType::parallel)
      .Case("reduction", IteratorType::reduction)
      .Default(::std::nullopt);
}
::std::optional<IteratorType> symbolizeIteratorType(uint32_t value) {
  switch (value) {
  case 0: return IteratorType::parallel;
  case 1: return IteratorType::reduction;
  default: return ::std::nullopt;
  }
}

} // namespace vector
} // namespace mlir

