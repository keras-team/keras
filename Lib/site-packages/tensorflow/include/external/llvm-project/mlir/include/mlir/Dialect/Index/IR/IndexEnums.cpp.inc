/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: IndexEnums.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace index {
::llvm::StringRef stringifyIndexCmpPredicate(IndexCmpPredicate val) {
  switch (val) {
    case IndexCmpPredicate::EQ: return "eq";
    case IndexCmpPredicate::NE: return "ne";
    case IndexCmpPredicate::SLT: return "slt";
    case IndexCmpPredicate::SLE: return "sle";
    case IndexCmpPredicate::SGT: return "sgt";
    case IndexCmpPredicate::SGE: return "sge";
    case IndexCmpPredicate::ULT: return "ult";
    case IndexCmpPredicate::ULE: return "ule";
    case IndexCmpPredicate::UGT: return "ugt";
    case IndexCmpPredicate::UGE: return "uge";
  }
  return "";
}

::std::optional<IndexCmpPredicate> symbolizeIndexCmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<IndexCmpPredicate>>(str)
      .Case("eq", IndexCmpPredicate::EQ)
      .Case("ne", IndexCmpPredicate::NE)
      .Case("slt", IndexCmpPredicate::SLT)
      .Case("sle", IndexCmpPredicate::SLE)
      .Case("sgt", IndexCmpPredicate::SGT)
      .Case("sge", IndexCmpPredicate::SGE)
      .Case("ult", IndexCmpPredicate::ULT)
      .Case("ule", IndexCmpPredicate::ULE)
      .Case("ugt", IndexCmpPredicate::UGT)
      .Case("uge", IndexCmpPredicate::UGE)
      .Default(::std::nullopt);
}
::std::optional<IndexCmpPredicate> symbolizeIndexCmpPredicate(uint32_t value) {
  switch (value) {
  case 0: return IndexCmpPredicate::EQ;
  case 1: return IndexCmpPredicate::NE;
  case 2: return IndexCmpPredicate::SLT;
  case 3: return IndexCmpPredicate::SLE;
  case 4: return IndexCmpPredicate::SGT;
  case 5: return IndexCmpPredicate::SGE;
  case 6: return IndexCmpPredicate::ULT;
  case 7: return IndexCmpPredicate::ULE;
  case 8: return IndexCmpPredicate::UGT;
  case 9: return IndexCmpPredicate::UGE;
  default: return ::std::nullopt;
  }
}

} // namespace index
} // namespace mlir

