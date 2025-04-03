/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: StructuredOpsUtils.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace utils {
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

} // namespace utils
} // namespace mlir

