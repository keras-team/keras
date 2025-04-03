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
namespace stablehlo {
class PrecisionAttr;
class FftTypeAttr;
class ComparisonDirectionAttr;
class ComparisonTypeAttr;
class TransposeAttr;
class RngDistributionAttr;
class RngAlgorithmAttr;
class ScatterDimensionNumbersAttr;
class GatherDimensionNumbersAttr;
class DotAlgorithmAttr;
class DotDimensionNumbersAttr;
class OutputOperandAliasAttr;
class ChannelHandleAttr;
class TypeExtensionsAttr;
class ConvDimensionNumbersAttr;
namespace detail {
struct PrecisionAttrStorage;
} // namespace detail
class PrecisionAttr : public ::mlir::Attribute::AttrBase<PrecisionAttr, ::mlir::Attribute, detail::PrecisionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.precision";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static PrecisionAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::Precision value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"precision"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::Precision getValue() const;
};
namespace detail {
struct FftTypeAttrStorage;
} // namespace detail
class FftTypeAttr : public ::mlir::Attribute::AttrBase<FftTypeAttr, ::mlir::Attribute, detail::FftTypeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.fft_type";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static FftTypeAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::FftType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fft_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::FftType getValue() const;
};
namespace detail {
struct ComparisonDirectionAttrStorage;
} // namespace detail
class ComparisonDirectionAttr : public ::mlir::Attribute::AttrBase<ComparisonDirectionAttr, ::mlir::Attribute, detail::ComparisonDirectionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.comparison_direction";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static ComparisonDirectionAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::ComparisonDirection value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"comparison_direction"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::ComparisonDirection getValue() const;
};
namespace detail {
struct ComparisonTypeAttrStorage;
} // namespace detail
class ComparisonTypeAttr : public ::mlir::Attribute::AttrBase<ComparisonTypeAttr, ::mlir::Attribute, detail::ComparisonTypeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.comparison_type";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static ComparisonTypeAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::ComparisonType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"comparison_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::ComparisonType getValue() const;
};
namespace detail {
struct TransposeAttrStorage;
} // namespace detail
class TransposeAttr : public ::mlir::Attribute::AttrBase<TransposeAttr, ::mlir::Attribute, detail::TransposeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.transpose";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static TransposeAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::Transpose value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"transpose"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::Transpose getValue() const;
};
namespace detail {
struct RngDistributionAttrStorage;
} // namespace detail
class RngDistributionAttr : public ::mlir::Attribute::AttrBase<RngDistributionAttr, ::mlir::Attribute, detail::RngDistributionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.rng_distribution";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static RngDistributionAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::RngDistribution value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"rng_distribution"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::RngDistribution getValue() const;
};
namespace detail {
struct RngAlgorithmAttrStorage;
} // namespace detail
class RngAlgorithmAttr : public ::mlir::Attribute::AttrBase<RngAlgorithmAttr, ::mlir::Attribute, detail::RngAlgorithmAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.rng_algorithm";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static RngAlgorithmAttr get(::mlir::MLIRContext *context, ::mlir::stablehlo::RngAlgorithm value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"rng_algorithm"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::stablehlo::RngAlgorithm getValue() const;
};
namespace detail {
struct ScatterDimensionNumbersAttrStorage;
} // namespace detail
class ScatterDimensionNumbersAttr : public ::mlir::Attribute::AttrBase<ScatterDimensionNumbersAttr, ::mlir::Attribute, detail::ScatterDimensionNumbersAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.scatter";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static ScatterDimensionNumbersAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> inputBatchingDims, ::llvm::ArrayRef<int64_t> scatterIndicesBatchingDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"scatter"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getUpdateWindowDims() const;
  ::llvm::ArrayRef<int64_t> getInsertedWindowDims() const;
  ::llvm::ArrayRef<int64_t> getInputBatchingDims() const;
  ::llvm::ArrayRef<int64_t> getScatterIndicesBatchingDims() const;
  ::llvm::ArrayRef<int64_t> getScatterDimsToOperandDims() const;
  int64_t getIndexVectorDim() const;
};
namespace detail {
struct GatherDimensionNumbersAttrStorage;
} // namespace detail
class GatherDimensionNumbersAttr : public ::mlir::Attribute::AttrBase<GatherDimensionNumbersAttr, ::mlir::Attribute, detail::GatherDimensionNumbersAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.gather";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static GatherDimensionNumbersAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> offsetDims, ::llvm::ArrayRef<int64_t> collapsedSliceDims, ::llvm::ArrayRef<int64_t> operandBatchingDims, ::llvm::ArrayRef<int64_t> startIndicesBatchingDims, ::llvm::ArrayRef<int64_t> startIndexMap, int64_t indexVectorDim);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"gather"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getOffsetDims() const;
  ::llvm::ArrayRef<int64_t> getCollapsedSliceDims() const;
  ::llvm::ArrayRef<int64_t> getOperandBatchingDims() const;
  ::llvm::ArrayRef<int64_t> getStartIndicesBatchingDims() const;
  ::llvm::ArrayRef<int64_t> getStartIndexMap() const;
  int64_t getIndexVectorDim() const;
};
namespace detail {
struct DotAlgorithmAttrStorage;
} // namespace detail
class DotAlgorithmAttr : public ::mlir::Attribute::AttrBase<DotAlgorithmAttr, ::mlir::Attribute, detail::DotAlgorithmAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.dot_algorithm";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  using Base::getChecked;
  static DotAlgorithmAttr get(::mlir::MLIRContext *context, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation);
  static DotAlgorithmAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dot_algorithm"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getLhsPrecisionType() const;
  Type getRhsPrecisionType() const;
  Type getAccumulationType() const;
  int64_t getLhsComponentCount() const;
  int64_t getRhsComponentCount() const;
  int64_t getNumPrimitiveOperations() const;
  bool getAllowImpreciseAccumulation() const;
};
namespace detail {
struct DotDimensionNumbersAttrStorage;
} // namespace detail
class DotDimensionNumbersAttr : public ::mlir::Attribute::AttrBase<DotDimensionNumbersAttr, ::mlir::Attribute, detail::DotDimensionNumbersAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.dot";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static DotDimensionNumbersAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> lhsBatchingDimensions, ::llvm::ArrayRef<int64_t> rhsBatchingDimensions, ::llvm::ArrayRef<int64_t> lhsContractingDimensions, ::llvm::ArrayRef<int64_t> rhsContractingDimensions);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dot"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getLhsBatchingDimensions() const;
  ::llvm::ArrayRef<int64_t> getRhsBatchingDimensions() const;
  ::llvm::ArrayRef<int64_t> getLhsContractingDimensions() const;
  ::llvm::ArrayRef<int64_t> getRhsContractingDimensions() const;
};
namespace detail {
struct OutputOperandAliasAttrStorage;
} // namespace detail
class OutputOperandAliasAttr : public ::mlir::Attribute::AttrBase<OutputOperandAliasAttr, ::mlir::Attribute, detail::OutputOperandAliasAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.output_operand_alias";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static OutputOperandAliasAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> outputTupleIndices, int64_t operandIndex, ::llvm::ArrayRef<int64_t> operandTupleIndices);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"output_operand_alias"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getOutputTupleIndices() const;
  int64_t getOperandIndex() const;
  ::llvm::ArrayRef<int64_t> getOperandTupleIndices() const;
};
namespace detail {
struct ChannelHandleAttrStorage;
} // namespace detail
class ChannelHandleAttr : public ::mlir::Attribute::AttrBase<ChannelHandleAttr, ::mlir::Attribute, detail::ChannelHandleAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.channel_handle";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static ChannelHandleAttr get(::mlir::MLIRContext *context, int64_t handle, int64_t type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"channel_handle"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getHandle() const;
  int64_t getType() const;
};
namespace detail {
struct TypeExtensionsAttrStorage;
} // namespace detail
class TypeExtensionsAttr : public ::mlir::Attribute::AttrBase<TypeExtensionsAttr, ::mlir::Attribute, detail::TypeExtensionsAttrStorage, ::mlir::VerifiableTensorEncoding::Trait, ::mlir::hlo::BoundedAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.type_extensions";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static TypeExtensionsAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> bounds);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"type_extensions"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getBounds() const;
  ::llvm::LogicalResult verifyEncoding(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
};
namespace detail {
struct ConvDimensionNumbersAttrStorage;
} // namespace detail
class ConvDimensionNumbersAttr : public ::mlir::Attribute::AttrBase<ConvDimensionNumbersAttr, ::mlir::Attribute, detail::ConvDimensionNumbersAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.conv";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static ConvDimensionNumbersAttr get(::mlir::MLIRContext *context, int64_t inputBatchDimension, int64_t inputFeatureDimension, ::llvm::ArrayRef<int64_t> inputSpatialDimensions, int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension, ::llvm::ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension, int64_t outputFeatureDimension, ::llvm::ArrayRef<int64_t> outputSpatialDimensions);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"conv"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputBatchDimension() const;
  int64_t getInputFeatureDimension() const;
  ::llvm::ArrayRef<int64_t> getInputSpatialDimensions() const;
  int64_t getKernelInputFeatureDimension() const;
  int64_t getKernelOutputFeatureDimension() const;
  ::llvm::ArrayRef<int64_t> getKernelSpatialDimensions() const;
  int64_t getOutputBatchDimension() const;
  int64_t getOutputFeatureDimension() const;
  ::llvm::ArrayRef<int64_t> getOutputSpatialDimensions() const;
};
} // namespace stablehlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::PrecisionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::FftTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ComparisonDirectionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ComparisonTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TransposeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::RngDistributionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::RngAlgorithmAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ScatterDimensionNumbersAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::GatherDimensionNumbersAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::DotAlgorithmAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::DotDimensionNumbersAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::OutputOperandAliasAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ChannelHandleAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TypeExtensionsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ConvDimensionNumbersAttr)

#endif  // GET_ATTRDEF_CLASSES

