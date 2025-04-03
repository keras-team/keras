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
namespace vhlo {
class ArrayV1Attr;
class BooleanV1Attr;
class DictionaryV1Attr;
class FloatV1Attr;
class IntegerV1Attr;
class OutputOperandAliasV1Attr;
class StringV1Attr;
class TensorV1Attr;
class TypeV1Attr;
class TypeExtensionsV1Attr;
class ComparisonDirectionV1Attr;
class ComparisonTypeV1Attr;
class CustomCallApiVersionV1Attr;
class FftTypeV1Attr;
class PrecisionV1Attr;
class RngAlgorithmV1Attr;
class RngDistributionV1Attr;
class TransposeV1Attr;
namespace detail {
struct ArrayV1AttrStorage;
} // namespace detail
class ArrayV1Attr : public ::mlir::Attribute::AttrBase<ArrayV1Attr, ::mlir::Attribute, detail::ArrayV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.array_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static ArrayV1Attr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<mlir::Attribute> value);
  static ArrayV1Attr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<mlir::Attribute> value);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<mlir::Attribute> value);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<mlir::Attribute> value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"array_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<mlir::Attribute> getValue() const;
};
namespace detail {
struct BooleanV1AttrStorage;
} // namespace detail
class BooleanV1Attr : public ::mlir::Attribute::AttrBase<BooleanV1Attr, ::mlir::Attribute, detail::BooleanV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.bool_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static BooleanV1Attr get(::mlir::MLIRContext *context, bool value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"bool_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  bool getValue() const;
};
namespace detail {
struct DictionaryV1AttrStorage;
} // namespace detail
class DictionaryV1Attr : public ::mlir::Attribute::AttrBase<DictionaryV1Attr, ::mlir::Attribute, detail::DictionaryV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.dict_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static DictionaryV1Attr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value);
  static DictionaryV1Attr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dict_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> getValue() const;
};
namespace detail {
struct FloatV1AttrStorage;
} // namespace detail
class FloatV1Attr : public ::mlir::Attribute::AttrBase<FloatV1Attr, ::mlir::Attribute, detail::FloatV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.float_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static FloatV1Attr get(::mlir::MLIRContext *context, mlir::Type type, ::llvm::APFloat value);
  static FloatV1Attr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, mlir::Type type, ::llvm::APFloat value);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, ::llvm::APFloat value);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, ::llvm::APFloat value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"float_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  mlir::Type getType() const;
  ::llvm::APFloat getValue() const;
};
namespace detail {
struct IntegerV1AttrStorage;
} // namespace detail
class IntegerV1Attr : public ::mlir::Attribute::AttrBase<IntegerV1Attr, ::mlir::Attribute, detail::IntegerV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.integer_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static IntegerV1Attr get(::mlir::MLIRContext *context, mlir::Type type, APInt value);
  static IntegerV1Attr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, mlir::Type type, APInt value);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, APInt value);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, APInt value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"integer_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  mlir::Type getType() const;
  APInt getValue() const;
};
namespace detail {
struct OutputOperandAliasV1AttrStorage;
} // namespace detail
class OutputOperandAliasV1Attr : public ::mlir::Attribute::AttrBase<OutputOperandAliasV1Attr, ::mlir::Attribute, detail::OutputOperandAliasV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.output_operand_alias_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static OutputOperandAliasV1Attr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> outputTupleIndices, int64_t operandIndex, ::llvm::ArrayRef<int64_t> operandTupleIndices);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"output_operand_alias_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getOutputTupleIndices() const;
  int64_t getOperandIndex() const;
  ::llvm::ArrayRef<int64_t> getOperandTupleIndices() const;
};
namespace detail {
struct StringV1AttrStorage;
} // namespace detail
class StringV1Attr : public ::mlir::Attribute::AttrBase<StringV1Attr, ::mlir::Attribute, detail::StringV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.string_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static StringV1Attr get(::mlir::MLIRContext *context, ::llvm::StringRef value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"string_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getValue() const;
};
namespace detail {
struct TensorV1AttrStorage;
} // namespace detail
class TensorV1Attr : public ::mlir::Attribute::AttrBase<TensorV1Attr, ::mlir::Attribute, detail::TensorV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.tensor_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static TensorV1Attr get(::mlir::MLIRContext *context, ::mlir::Type type, ::llvm::ArrayRef<char> data);
  static TensorV1Attr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type type, ::llvm::ArrayRef<char> data);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, ::llvm::ArrayRef<char> data);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, ::llvm::ArrayRef<char> data);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::Type getType() const;
  ::llvm::ArrayRef<char> getData() const;
};
namespace detail {
struct TypeV1AttrStorage;
} // namespace detail
class TypeV1Attr : public ::mlir::Attribute::AttrBase<TypeV1Attr, ::mlir::Attribute, detail::TypeV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.type_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  using Base::getChecked;
  static TypeV1Attr get(::mlir::MLIRContext *context, ::mlir::Type value);
  static TypeV1Attr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type value);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type value);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"type_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::Type getValue() const;
};
namespace detail {
struct TypeExtensionsV1AttrStorage;
} // namespace detail
class TypeExtensionsV1Attr : public ::mlir::Attribute::AttrBase<TypeExtensionsV1Attr, ::mlir::Attribute, detail::TypeExtensionsV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.type_extensions_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static TypeExtensionsV1Attr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> bounds);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"type_extensions_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getBounds() const;
};
namespace detail {
struct ComparisonDirectionV1AttrStorage;
} // namespace detail
class ComparisonDirectionV1Attr : public ::mlir::Attribute::AttrBase<ComparisonDirectionV1Attr, ::mlir::Attribute, detail::ComparisonDirectionV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.comparison_direction_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static ComparisonDirectionV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::ComparisonDirectionV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"comparison_direction_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::ComparisonDirectionV1 getValue() const;
};
namespace detail {
struct ComparisonTypeV1AttrStorage;
} // namespace detail
class ComparisonTypeV1Attr : public ::mlir::Attribute::AttrBase<ComparisonTypeV1Attr, ::mlir::Attribute, detail::ComparisonTypeV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.comparison_type_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static ComparisonTypeV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::ComparisonTypeV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"comparison_type_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::ComparisonTypeV1 getValue() const;
};
namespace detail {
struct CustomCallApiVersionV1AttrStorage;
} // namespace detail
class CustomCallApiVersionV1Attr : public ::mlir::Attribute::AttrBase<CustomCallApiVersionV1Attr, ::mlir::Attribute, detail::CustomCallApiVersionV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.api_version_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static CustomCallApiVersionV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::CustomCallApiVersionV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"api_version_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::CustomCallApiVersionV1 getValue() const;
};
namespace detail {
struct FftTypeV1AttrStorage;
} // namespace detail
class FftTypeV1Attr : public ::mlir::Attribute::AttrBase<FftTypeV1Attr, ::mlir::Attribute, detail::FftTypeV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.fft_type_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static FftTypeV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::FftTypeV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fft_type_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::FftTypeV1 getValue() const;
};
namespace detail {
struct PrecisionV1AttrStorage;
} // namespace detail
class PrecisionV1Attr : public ::mlir::Attribute::AttrBase<PrecisionV1Attr, ::mlir::Attribute, detail::PrecisionV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.precision_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static PrecisionV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::PrecisionV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"precision_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::PrecisionV1 getValue() const;
};
namespace detail {
struct RngAlgorithmV1AttrStorage;
} // namespace detail
class RngAlgorithmV1Attr : public ::mlir::Attribute::AttrBase<RngAlgorithmV1Attr, ::mlir::Attribute, detail::RngAlgorithmV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.rng_algorithm_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static RngAlgorithmV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::RngAlgorithmV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"rng_algorithm_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::RngAlgorithmV1 getValue() const;
};
namespace detail {
struct RngDistributionV1AttrStorage;
} // namespace detail
class RngDistributionV1Attr : public ::mlir::Attribute::AttrBase<RngDistributionV1Attr, ::mlir::Attribute, detail::RngDistributionV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.rng_distribution_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static RngDistributionV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::RngDistributionV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"rng_distribution_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::RngDistributionV1 getValue() const;
};
namespace detail {
struct TransposeV1AttrStorage;
} // namespace detail
class TransposeV1Attr : public ::mlir::Attribute::AttrBase<TransposeV1Attr, ::mlir::Attribute, detail::TransposeV1AttrStorage, ::mlir::vhlo::VersionedAttrInterface::Trait> {
public:
  using Base::Base;
  mlir::vhlo::Version getMinVersion() {
    return mlir::vhlo::Version(0, 9, 0);
  }
  mlir::vhlo::Version getMaxVersion() {
     return mlir::vhlo::Version::getCurrentVersion(); 
  }
  static constexpr ::llvm::StringLiteral name = "vhlo.transpose_v1";
  static constexpr ::llvm::StringLiteral dialectName = "vhlo";
  static TransposeV1Attr get(::mlir::MLIRContext *context, ::mlir::vhlo::TransposeV1 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"transpose_v1"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vhlo::TransposeV1 getValue() const;
};
} // namespace vhlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::ArrayV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::BooleanV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::DictionaryV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::OutputOperandAliasV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::StringV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::TensorV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::TypeV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::TypeExtensionsV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::ComparisonDirectionV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::ComparisonTypeV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::CustomCallApiVersionV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::FftTypeV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::PrecisionV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::RngAlgorithmV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::RngDistributionV1Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vhlo::TransposeV1Attr)

#endif  // GET_ATTRDEF_CLASSES

