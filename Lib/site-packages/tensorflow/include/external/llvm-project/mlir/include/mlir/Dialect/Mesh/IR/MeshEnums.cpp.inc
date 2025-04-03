/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MeshOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mesh {
::llvm::StringRef stringifyReductionKind(ReductionKind val) {
  switch (val) {
    case ReductionKind::Sum: return "sum";
    case ReductionKind::Max: return "max";
    case ReductionKind::Min: return "min";
    case ReductionKind::Product: return "product";
    case ReductionKind::Average: return "average";
    case ReductionKind::BitwiseAnd: return "bitwise_and";
    case ReductionKind::BitwiseOr: return "bitwise_or";
    case ReductionKind::BitwiseXor: return "bitwise_xor";
    case ReductionKind::Generic: return "generic";
  }
  return "";
}

::std::optional<ReductionKind> symbolizeReductionKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ReductionKind>>(str)
      .Case("sum", ReductionKind::Sum)
      .Case("max", ReductionKind::Max)
      .Case("min", ReductionKind::Min)
      .Case("product", ReductionKind::Product)
      .Case("average", ReductionKind::Average)
      .Case("bitwise_and", ReductionKind::BitwiseAnd)
      .Case("bitwise_or", ReductionKind::BitwiseOr)
      .Case("bitwise_xor", ReductionKind::BitwiseXor)
      .Case("generic", ReductionKind::Generic)
      .Default(::std::nullopt);
}
::std::optional<ReductionKind> symbolizeReductionKind(uint32_t value) {
  switch (value) {
  case 1: return ReductionKind::Sum;
  case 2: return ReductionKind::Max;
  case 3: return ReductionKind::Min;
  case 4: return ReductionKind::Product;
  case 5: return ReductionKind::Average;
  case 6: return ReductionKind::BitwiseAnd;
  case 7: return ReductionKind::BitwiseOr;
  case 8: return ReductionKind::BitwiseXor;
  case 100: return ReductionKind::Generic;
  default: return ::std::nullopt;
  }
}

} // namespace mesh
} // namespace mlir

