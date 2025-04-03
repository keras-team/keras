/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace vector {
class CombiningKindAttr;
class IteratorTypeAttr;
class PrintPunctuationAttr;
namespace detail {
struct CombiningKindAttrStorage;
} // namespace detail
class CombiningKindAttr : public ::mlir::Attribute::AttrBase<CombiningKindAttr, ::mlir::Attribute, detail::CombiningKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "vector.kind";
  static constexpr ::llvm::StringLiteral dialectName = "vector";
  static CombiningKindAttr get(::mlir::MLIRContext *context, ::mlir::vector::CombiningKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vector::CombiningKind getValue() const;
};
namespace detail {
struct IteratorTypeAttrStorage;
} // namespace detail
class IteratorTypeAttr : public ::mlir::Attribute::AttrBase<IteratorTypeAttr, ::mlir::Attribute, detail::IteratorTypeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "vector.iterator_type";
  static constexpr ::llvm::StringLiteral dialectName = "vector";
  static IteratorTypeAttr get(::mlir::MLIRContext *context, ::mlir::vector::IteratorType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"iterator_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vector::IteratorType getValue() const;
};
namespace detail {
struct PrintPunctuationAttrStorage;
} // namespace detail
class PrintPunctuationAttr : public ::mlir::Attribute::AttrBase<PrintPunctuationAttr, ::mlir::Attribute, detail::PrintPunctuationAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "vector.punctuation";
  static constexpr ::llvm::StringLiteral dialectName = "vector";
  static PrintPunctuationAttr get(::mlir::MLIRContext *context, ::mlir::vector::PrintPunctuation value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"punctuation"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vector::PrintPunctuation getValue() const;
};
} // namespace vector
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::CombiningKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::IteratorTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::PrintPunctuationAttr)

#endif  // GET_ATTRDEF_CLASSES

