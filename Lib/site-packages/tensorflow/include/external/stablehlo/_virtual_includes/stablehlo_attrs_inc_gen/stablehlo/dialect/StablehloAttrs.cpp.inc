/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::stablehlo::PrecisionAttr,
::mlir::stablehlo::FftTypeAttr,
::mlir::stablehlo::ComparisonDirectionAttr,
::mlir::stablehlo::ComparisonTypeAttr,
::mlir::stablehlo::TransposeAttr,
::mlir::stablehlo::RngDistributionAttr,
::mlir::stablehlo::RngAlgorithmAttr,
::mlir::stablehlo::ScatterDimensionNumbersAttr,
::mlir::stablehlo::GatherDimensionNumbersAttr,
::mlir::stablehlo::DotAlgorithmAttr,
::mlir::stablehlo::DotDimensionNumbersAttr,
::mlir::stablehlo::OutputOperandAliasAttr,
::mlir::stablehlo::ChannelHandleAttr,
::mlir::stablehlo::TypeExtensionsAttr,
::mlir::stablehlo::ConvDimensionNumbersAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::stablehlo::PrecisionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::PrecisionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::FftTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::FftTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ComparisonDirectionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ComparisonDirectionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ComparisonTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ComparisonTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::TransposeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::TransposeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::RngDistributionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::RngDistributionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::RngAlgorithmAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::RngAlgorithmAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ScatterDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ScatterDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::GatherDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::GatherDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::DotAlgorithmAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::DotAlgorithmAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::DotDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::DotDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::OutputOperandAliasAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::OutputOperandAliasAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ChannelHandleAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ChannelHandleAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::TypeExtensionsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::TypeExtensionsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ConvDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ConvDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::stablehlo::PrecisionAttr>([&](auto t) {
      printer << ::mlir::stablehlo::PrecisionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::FftTypeAttr>([&](auto t) {
      printer << ::mlir::stablehlo::FftTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ComparisonDirectionAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ComparisonDirectionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ComparisonTypeAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ComparisonTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::TransposeAttr>([&](auto t) {
      printer << ::mlir::stablehlo::TransposeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::RngDistributionAttr>([&](auto t) {
      printer << ::mlir::stablehlo::RngDistributionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::RngAlgorithmAttr>([&](auto t) {
      printer << ::mlir::stablehlo::RngAlgorithmAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ScatterDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ScatterDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::GatherDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::GatherDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::DotAlgorithmAttr>([&](auto t) {
      printer << ::mlir::stablehlo::DotAlgorithmAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::DotDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::DotDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::OutputOperandAliasAttr>([&](auto t) {
      printer << ::mlir::stablehlo::OutputOperandAliasAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ChannelHandleAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ChannelHandleAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::TypeExtensionsAttr>([&](auto t) {
      printer << ::mlir::stablehlo::TypeExtensionsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ConvDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ConvDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace stablehlo {
namespace detail {
struct PrecisionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::Precision>;
  PrecisionAttrStorage(::mlir::stablehlo::Precision value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PrecisionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PrecisionAttrStorage>()) PrecisionAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::Precision value;
};
} // namespace detail
PrecisionAttr PrecisionAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::Precision value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute PrecisionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::Precision> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::Precision> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizePrecision(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::Precision" << " to be one of: " << "DEFAULT" << ", " << "HIGH" << ", " << "HIGHEST")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_PrecisionAttr parameter 'value' which is to be a `::mlir::stablehlo::Precision`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return PrecisionAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::Precision((*_result_value)));
}

void PrecisionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyPrecision(getValue());
}

::mlir::stablehlo::Precision PrecisionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::PrecisionAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct FftTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::FftType>;
  FftTypeAttrStorage(::mlir::stablehlo::FftType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FftTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<FftTypeAttrStorage>()) FftTypeAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::FftType value;
};
} // namespace detail
FftTypeAttr FftTypeAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::FftType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute FftTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::FftType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::FftType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeFftType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::FftType" << " to be one of: " << "FFT" << ", " << "IFFT" << ", " << "RFFT" << ", " << "IRFFT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_FftTypeAttr parameter 'value' which is to be a `::mlir::stablehlo::FftType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return FftTypeAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::FftType((*_result_value)));
}

void FftTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyFftType(getValue());
}

::mlir::stablehlo::FftType FftTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::FftTypeAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ComparisonDirectionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::ComparisonDirection>;
  ComparisonDirectionAttrStorage(::mlir::stablehlo::ComparisonDirection value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonDirectionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComparisonDirectionAttrStorage>()) ComparisonDirectionAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::ComparisonDirection value;
};
} // namespace detail
ComparisonDirectionAttr ComparisonDirectionAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::ComparisonDirection value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ComparisonDirectionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::ComparisonDirection> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::ComparisonDirection> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeComparisonDirection(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::ComparisonDirection" << " to be one of: " << "EQ" << ", " << "NE" << ", " << "GE" << ", " << "GT" << ", " << "LE" << ", " << "LT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ComparisonDirectionAttr parameter 'value' which is to be a `::mlir::stablehlo::ComparisonDirection`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonDirectionAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::ComparisonDirection((*_result_value)));
}

void ComparisonDirectionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonDirection(getValue());
}

::mlir::stablehlo::ComparisonDirection ComparisonDirectionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ComparisonDirectionAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ComparisonTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::ComparisonType>;
  ComparisonTypeAttrStorage(::mlir::stablehlo::ComparisonType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComparisonTypeAttrStorage>()) ComparisonTypeAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::ComparisonType value;
};
} // namespace detail
ComparisonTypeAttr ComparisonTypeAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::ComparisonType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ComparisonTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::ComparisonType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::ComparisonType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeComparisonType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::ComparisonType" << " to be one of: " << "NOTYPE" << ", " << "FLOAT" << ", " << "TOTALORDER" << ", " << "SIGNED" << ", " << "UNSIGNED")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ComparisonTypeAttr parameter 'value' which is to be a `::mlir::stablehlo::ComparisonType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonTypeAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::ComparisonType((*_result_value)));
}

void ComparisonTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonType(getValue());
}

::mlir::stablehlo::ComparisonType ComparisonTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ComparisonTypeAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct TransposeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::Transpose>;
  TransposeAttrStorage(::mlir::stablehlo::Transpose value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TransposeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TransposeAttrStorage>()) TransposeAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::Transpose value;
};
} // namespace detail
TransposeAttr TransposeAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::Transpose value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TransposeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::Transpose> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::Transpose> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeTranspose(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::Transpose" << " to be one of: " << "TRANSPOSE_INVALID" << ", " << "NO_TRANSPOSE" << ", " << "TRANSPOSE" << ", " << "ADJOINT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_TransposeAttr parameter 'value' which is to be a `::mlir::stablehlo::Transpose`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TransposeAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::Transpose((*_result_value)));
}

void TransposeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTranspose(getValue());
}

::mlir::stablehlo::Transpose TransposeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TransposeAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct RngDistributionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::RngDistribution>;
  RngDistributionAttrStorage(::mlir::stablehlo::RngDistribution value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RngDistributionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RngDistributionAttrStorage>()) RngDistributionAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::RngDistribution value;
};
} // namespace detail
RngDistributionAttr RngDistributionAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::RngDistribution value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RngDistributionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::RngDistribution> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::RngDistribution> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeRngDistribution(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::RngDistribution" << " to be one of: " << "UNIFORM" << ", " << "NORMAL")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_RngDistributionAttr parameter 'value' which is to be a `::mlir::stablehlo::RngDistribution`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RngDistributionAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::RngDistribution((*_result_value)));
}

void RngDistributionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRngDistribution(getValue());
}

::mlir::stablehlo::RngDistribution RngDistributionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::RngDistributionAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct RngAlgorithmAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::RngAlgorithm>;
  RngAlgorithmAttrStorage(::mlir::stablehlo::RngAlgorithm value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RngAlgorithmAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RngAlgorithmAttrStorage>()) RngAlgorithmAttrStorage(std::move(value));
  }

  ::mlir::stablehlo::RngAlgorithm value;
};
} // namespace detail
RngAlgorithmAttr RngAlgorithmAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::RngAlgorithm value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RngAlgorithmAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::RngAlgorithm> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::RngAlgorithm> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeRngAlgorithm(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::RngAlgorithm" << " to be one of: " << "DEFAULT" << ", " << "THREE_FRY" << ", " << "PHILOX")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_RngAlgorithmAttr parameter 'value' which is to be a `::mlir::stablehlo::RngAlgorithm`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RngAlgorithmAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::RngAlgorithm((*_result_value)));
}

void RngAlgorithmAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRngAlgorithm(getValue());
}

::mlir::stablehlo::RngAlgorithm RngAlgorithmAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::RngAlgorithmAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ScatterDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, int64_t>;
  ScatterDimensionNumbersAttrStorage(::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> inputBatchingDims, ::llvm::ArrayRef<int64_t> scatterIndicesBatchingDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim) : updateWindowDims(std::move(updateWindowDims)), insertedWindowDims(std::move(insertedWindowDims)), inputBatchingDims(std::move(inputBatchingDims)), scatterIndicesBatchingDims(std::move(scatterIndicesBatchingDims)), scatterDimsToOperandDims(std::move(scatterDimsToOperandDims)), indexVectorDim(std::move(indexVectorDim)) {}

  KeyTy getAsKey() const {
    return KeyTy(updateWindowDims, insertedWindowDims, inputBatchingDims, scatterIndicesBatchingDims, scatterDimsToOperandDims, indexVectorDim);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (updateWindowDims == std::get<0>(tblgenKey)) && (insertedWindowDims == std::get<1>(tblgenKey)) && (inputBatchingDims == std::get<2>(tblgenKey)) && (scatterIndicesBatchingDims == std::get<3>(tblgenKey)) && (scatterDimsToOperandDims == std::get<4>(tblgenKey)) && (indexVectorDim == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static ScatterDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto updateWindowDims = std::move(std::get<0>(tblgenKey));
    auto insertedWindowDims = std::move(std::get<1>(tblgenKey));
    auto inputBatchingDims = std::move(std::get<2>(tblgenKey));
    auto scatterIndicesBatchingDims = std::move(std::get<3>(tblgenKey));
    auto scatterDimsToOperandDims = std::move(std::get<4>(tblgenKey));
    auto indexVectorDim = std::move(std::get<5>(tblgenKey));
    updateWindowDims = allocator.copyInto(updateWindowDims);
    insertedWindowDims = allocator.copyInto(insertedWindowDims);
    inputBatchingDims = allocator.copyInto(inputBatchingDims);
    scatterIndicesBatchingDims = allocator.copyInto(scatterIndicesBatchingDims);
    scatterDimsToOperandDims = allocator.copyInto(scatterDimsToOperandDims);
    return new (allocator.allocate<ScatterDimensionNumbersAttrStorage>()) ScatterDimensionNumbersAttrStorage(std::move(updateWindowDims), std::move(insertedWindowDims), std::move(inputBatchingDims), std::move(scatterIndicesBatchingDims), std::move(scatterDimsToOperandDims), std::move(indexVectorDim));
  }

  ::llvm::ArrayRef<int64_t> updateWindowDims;
  ::llvm::ArrayRef<int64_t> insertedWindowDims;
  ::llvm::ArrayRef<int64_t> inputBatchingDims;
  ::llvm::ArrayRef<int64_t> scatterIndicesBatchingDims;
  ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim;
};
} // namespace detail
ScatterDimensionNumbersAttr ScatterDimensionNumbersAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> inputBatchingDims, ::llvm::ArrayRef<int64_t> scatterIndicesBatchingDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim) {
  return Base::get(context, std::move(updateWindowDims), std::move(insertedWindowDims), std::move(inputBatchingDims), std::move(scatterIndicesBatchingDims), std::move(scatterDimsToOperandDims), std::move(indexVectorDim));
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getUpdateWindowDims() const {
  return getImpl()->updateWindowDims;
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getInsertedWindowDims() const {
  return getImpl()->insertedWindowDims;
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getInputBatchingDims() const {
  return getImpl()->inputBatchingDims;
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getScatterIndicesBatchingDims() const {
  return getImpl()->scatterIndicesBatchingDims;
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getScatterDimsToOperandDims() const {
  return getImpl()->scatterDimsToOperandDims;
}

int64_t ScatterDimensionNumbersAttr::getIndexVectorDim() const {
  return getImpl()->indexVectorDim;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ScatterDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct GatherDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, int64_t>;
  GatherDimensionNumbersAttrStorage(::llvm::ArrayRef<int64_t> offsetDims, ::llvm::ArrayRef<int64_t> collapsedSliceDims, ::llvm::ArrayRef<int64_t> operandBatchingDims, ::llvm::ArrayRef<int64_t> startIndicesBatchingDims, ::llvm::ArrayRef<int64_t> startIndexMap, int64_t indexVectorDim) : offsetDims(std::move(offsetDims)), collapsedSliceDims(std::move(collapsedSliceDims)), operandBatchingDims(std::move(operandBatchingDims)), startIndicesBatchingDims(std::move(startIndicesBatchingDims)), startIndexMap(std::move(startIndexMap)), indexVectorDim(std::move(indexVectorDim)) {}

  KeyTy getAsKey() const {
    return KeyTy(offsetDims, collapsedSliceDims, operandBatchingDims, startIndicesBatchingDims, startIndexMap, indexVectorDim);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (offsetDims == std::get<0>(tblgenKey)) && (collapsedSliceDims == std::get<1>(tblgenKey)) && (operandBatchingDims == std::get<2>(tblgenKey)) && (startIndicesBatchingDims == std::get<3>(tblgenKey)) && (startIndexMap == std::get<4>(tblgenKey)) && (indexVectorDim == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static GatherDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto offsetDims = std::move(std::get<0>(tblgenKey));
    auto collapsedSliceDims = std::move(std::get<1>(tblgenKey));
    auto operandBatchingDims = std::move(std::get<2>(tblgenKey));
    auto startIndicesBatchingDims = std::move(std::get<3>(tblgenKey));
    auto startIndexMap = std::move(std::get<4>(tblgenKey));
    auto indexVectorDim = std::move(std::get<5>(tblgenKey));
    offsetDims = allocator.copyInto(offsetDims);
    collapsedSliceDims = allocator.copyInto(collapsedSliceDims);
    operandBatchingDims = allocator.copyInto(operandBatchingDims);
    startIndicesBatchingDims = allocator.copyInto(startIndicesBatchingDims);
    startIndexMap = allocator.copyInto(startIndexMap);
    return new (allocator.allocate<GatherDimensionNumbersAttrStorage>()) GatherDimensionNumbersAttrStorage(std::move(offsetDims), std::move(collapsedSliceDims), std::move(operandBatchingDims), std::move(startIndicesBatchingDims), std::move(startIndexMap), std::move(indexVectorDim));
  }

  ::llvm::ArrayRef<int64_t> offsetDims;
  ::llvm::ArrayRef<int64_t> collapsedSliceDims;
  ::llvm::ArrayRef<int64_t> operandBatchingDims;
  ::llvm::ArrayRef<int64_t> startIndicesBatchingDims;
  ::llvm::ArrayRef<int64_t> startIndexMap;
  int64_t indexVectorDim;
};
} // namespace detail
GatherDimensionNumbersAttr GatherDimensionNumbersAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> offsetDims, ::llvm::ArrayRef<int64_t> collapsedSliceDims, ::llvm::ArrayRef<int64_t> operandBatchingDims, ::llvm::ArrayRef<int64_t> startIndicesBatchingDims, ::llvm::ArrayRef<int64_t> startIndexMap, int64_t indexVectorDim) {
  return Base::get(context, std::move(offsetDims), std::move(collapsedSliceDims), std::move(operandBatchingDims), std::move(startIndicesBatchingDims), std::move(startIndexMap), std::move(indexVectorDim));
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getOffsetDims() const {
  return getImpl()->offsetDims;
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getCollapsedSliceDims() const {
  return getImpl()->collapsedSliceDims;
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getOperandBatchingDims() const {
  return getImpl()->operandBatchingDims;
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getStartIndicesBatchingDims() const {
  return getImpl()->startIndicesBatchingDims;
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getStartIndexMap() const {
  return getImpl()->startIndexMap;
}

int64_t GatherDimensionNumbersAttr::getIndexVectorDim() const {
  return getImpl()->indexVectorDim;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::GatherDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct DotAlgorithmAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Type, Type, Type, int64_t, int64_t, int64_t, bool>;
  DotAlgorithmAttrStorage(Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation) : lhsPrecisionType(std::move(lhsPrecisionType)), rhsPrecisionType(std::move(rhsPrecisionType)), accumulationType(std::move(accumulationType)), lhsComponentCount(std::move(lhsComponentCount)), rhsComponentCount(std::move(rhsComponentCount)), numPrimitiveOperations(std::move(numPrimitiveOperations)), allowImpreciseAccumulation(std::move(allowImpreciseAccumulation)) {}

  KeyTy getAsKey() const {
    return KeyTy(lhsPrecisionType, rhsPrecisionType, accumulationType, lhsComponentCount, rhsComponentCount, numPrimitiveOperations, allowImpreciseAccumulation);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (lhsPrecisionType == std::get<0>(tblgenKey)) && (rhsPrecisionType == std::get<1>(tblgenKey)) && (accumulationType == std::get<2>(tblgenKey)) && (lhsComponentCount == std::get<3>(tblgenKey)) && (rhsComponentCount == std::get<4>(tblgenKey)) && (numPrimitiveOperations == std::get<5>(tblgenKey)) && (allowImpreciseAccumulation == std::get<6>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
  }

  static DotAlgorithmAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto lhsPrecisionType = std::move(std::get<0>(tblgenKey));
    auto rhsPrecisionType = std::move(std::get<1>(tblgenKey));
    auto accumulationType = std::move(std::get<2>(tblgenKey));
    auto lhsComponentCount = std::move(std::get<3>(tblgenKey));
    auto rhsComponentCount = std::move(std::get<4>(tblgenKey));
    auto numPrimitiveOperations = std::move(std::get<5>(tblgenKey));
    auto allowImpreciseAccumulation = std::move(std::get<6>(tblgenKey));
    return new (allocator.allocate<DotAlgorithmAttrStorage>()) DotAlgorithmAttrStorage(std::move(lhsPrecisionType), std::move(rhsPrecisionType), std::move(accumulationType), std::move(lhsComponentCount), std::move(rhsComponentCount), std::move(numPrimitiveOperations), std::move(allowImpreciseAccumulation));
  }

  Type lhsPrecisionType;
  Type rhsPrecisionType;
  Type accumulationType;
  int64_t lhsComponentCount;
  int64_t rhsComponentCount;
  int64_t numPrimitiveOperations;
  bool allowImpreciseAccumulation;
};
} // namespace detail
DotAlgorithmAttr DotAlgorithmAttr::get(::mlir::MLIRContext *context, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation) {
  return Base::get(context, std::move(lhsPrecisionType), std::move(rhsPrecisionType), std::move(accumulationType), std::move(lhsComponentCount), std::move(rhsComponentCount), std::move(numPrimitiveOperations), std::move(allowImpreciseAccumulation));
}

DotAlgorithmAttr DotAlgorithmAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation) {
  return Base::getChecked(emitError, context, lhsPrecisionType, rhsPrecisionType, accumulationType, lhsComponentCount, rhsComponentCount, numPrimitiveOperations, allowImpreciseAccumulation);
}

::llvm::LogicalResult DotAlgorithmAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount, int64_t rhsComponentCount, int64_t numPrimitiveOperations, bool allowImpreciseAccumulation) {
  if (::mlir::failed(verify(emitError, lhsPrecisionType, rhsPrecisionType, accumulationType, lhsComponentCount, rhsComponentCount, numPrimitiveOperations, allowImpreciseAccumulation)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute DotAlgorithmAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_lhsPrecisionType;
  ::mlir::FailureOr<Type> _result_rhsPrecisionType;
  ::mlir::FailureOr<Type> _result_accumulationType;
  ::mlir::FailureOr<int64_t> _result_lhsComponentCount;
  ::mlir::FailureOr<int64_t> _result_rhsComponentCount;
  ::mlir::FailureOr<int64_t> _result_numPrimitiveOperations;
  ::mlir::FailureOr<bool> _result_allowImpreciseAccumulation;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'lhs_precision_type'
  if (odsParser.parseKeyword("lhs_precision_type")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'lhsPrecisionType'
  _result_lhsPrecisionType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_lhsPrecisionType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'lhsPrecisionType' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'rhs_precision_type'
  if (odsParser.parseKeyword("rhs_precision_type")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'rhsPrecisionType'
  _result_rhsPrecisionType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_rhsPrecisionType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'rhsPrecisionType' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'accumulation_type'
  if (odsParser.parseKeyword("accumulation_type")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'accumulationType'
  _result_accumulationType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_accumulationType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'accumulationType' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'lhs_component_count'
  if (odsParser.parseKeyword("lhs_component_count")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'lhsComponentCount'
  _result_lhsComponentCount = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_lhsComponentCount)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'lhsComponentCount' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'rhs_component_count'
  if (odsParser.parseKeyword("rhs_component_count")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'rhsComponentCount'
  _result_rhsComponentCount = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_rhsComponentCount)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'rhsComponentCount' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'num_primitive_operations'
  if (odsParser.parseKeyword("num_primitive_operations")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'numPrimitiveOperations'
  _result_numPrimitiveOperations = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_numPrimitiveOperations)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'numPrimitiveOperations' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'allow_imprecise_accumulation'
  if (odsParser.parseKeyword("allow_imprecise_accumulation")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'allowImpreciseAccumulation'
  _result_allowImpreciseAccumulation = ::mlir::FieldParser<bool>::parse(odsParser);
  if (::mlir::failed(_result_allowImpreciseAccumulation)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_DotAlgorithm parameter 'allowImpreciseAccumulation' which is to be a `bool`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_lhsPrecisionType));
  assert(::mlir::succeeded(_result_rhsPrecisionType));
  assert(::mlir::succeeded(_result_accumulationType));
  assert(::mlir::succeeded(_result_lhsComponentCount));
  assert(::mlir::succeeded(_result_rhsComponentCount));
  assert(::mlir::succeeded(_result_numPrimitiveOperations));
  assert(::mlir::succeeded(_result_allowImpreciseAccumulation));
  return odsParser.getChecked<DotAlgorithmAttr>(odsLoc, odsParser.getContext(),
      Type((*_result_lhsPrecisionType)),
      Type((*_result_rhsPrecisionType)),
      Type((*_result_accumulationType)),
      int64_t((*_result_lhsComponentCount)),
      int64_t((*_result_rhsComponentCount)),
      int64_t((*_result_numPrimitiveOperations)),
      bool((*_result_allowImpreciseAccumulation)));
}

void DotAlgorithmAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "lhs_precision_type";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLhsPrecisionType());
  odsPrinter << ",";
  odsPrinter << ' ' << "rhs_precision_type";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getRhsPrecisionType());
  odsPrinter << ",";
  odsPrinter << ' ' << "accumulation_type";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getAccumulationType());
  odsPrinter << ",";
  odsPrinter << ' ' << "lhs_component_count";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLhsComponentCount());
  odsPrinter << ",";
  odsPrinter << ' ' << "rhs_component_count";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getRhsComponentCount());
  odsPrinter << ",";
  odsPrinter << ' ' << "num_primitive_operations";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getNumPrimitiveOperations());
  odsPrinter << ",";
  odsPrinter << ' ' << "allow_imprecise_accumulation";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getAllowImpreciseAccumulation());
  odsPrinter << ">";
}

Type DotAlgorithmAttr::getLhsPrecisionType() const {
  return getImpl()->lhsPrecisionType;
}

Type DotAlgorithmAttr::getRhsPrecisionType() const {
  return getImpl()->rhsPrecisionType;
}

Type DotAlgorithmAttr::getAccumulationType() const {
  return getImpl()->accumulationType;
}

int64_t DotAlgorithmAttr::getLhsComponentCount() const {
  return getImpl()->lhsComponentCount;
}

int64_t DotAlgorithmAttr::getRhsComponentCount() const {
  return getImpl()->rhsComponentCount;
}

int64_t DotAlgorithmAttr::getNumPrimitiveOperations() const {
  return getImpl()->numPrimitiveOperations;
}

bool DotAlgorithmAttr::getAllowImpreciseAccumulation() const {
  return getImpl()->allowImpreciseAccumulation;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::DotAlgorithmAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct DotDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>>;
  DotDimensionNumbersAttrStorage(::llvm::ArrayRef<int64_t> lhsBatchingDimensions, ::llvm::ArrayRef<int64_t> rhsBatchingDimensions, ::llvm::ArrayRef<int64_t> lhsContractingDimensions, ::llvm::ArrayRef<int64_t> rhsContractingDimensions) : lhsBatchingDimensions(std::move(lhsBatchingDimensions)), rhsBatchingDimensions(std::move(rhsBatchingDimensions)), lhsContractingDimensions(std::move(lhsContractingDimensions)), rhsContractingDimensions(std::move(rhsContractingDimensions)) {}

  KeyTy getAsKey() const {
    return KeyTy(lhsBatchingDimensions, rhsBatchingDimensions, lhsContractingDimensions, rhsContractingDimensions);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (lhsBatchingDimensions == std::get<0>(tblgenKey)) && (rhsBatchingDimensions == std::get<1>(tblgenKey)) && (lhsContractingDimensions == std::get<2>(tblgenKey)) && (rhsContractingDimensions == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static DotDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto lhsBatchingDimensions = std::move(std::get<0>(tblgenKey));
    auto rhsBatchingDimensions = std::move(std::get<1>(tblgenKey));
    auto lhsContractingDimensions = std::move(std::get<2>(tblgenKey));
    auto rhsContractingDimensions = std::move(std::get<3>(tblgenKey));
    lhsBatchingDimensions = allocator.copyInto(lhsBatchingDimensions);
    rhsBatchingDimensions = allocator.copyInto(rhsBatchingDimensions);
    lhsContractingDimensions = allocator.copyInto(lhsContractingDimensions);
    rhsContractingDimensions = allocator.copyInto(rhsContractingDimensions);
    return new (allocator.allocate<DotDimensionNumbersAttrStorage>()) DotDimensionNumbersAttrStorage(std::move(lhsBatchingDimensions), std::move(rhsBatchingDimensions), std::move(lhsContractingDimensions), std::move(rhsContractingDimensions));
  }

  ::llvm::ArrayRef<int64_t> lhsBatchingDimensions;
  ::llvm::ArrayRef<int64_t> rhsBatchingDimensions;
  ::llvm::ArrayRef<int64_t> lhsContractingDimensions;
  ::llvm::ArrayRef<int64_t> rhsContractingDimensions;
};
} // namespace detail
DotDimensionNumbersAttr DotDimensionNumbersAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> lhsBatchingDimensions, ::llvm::ArrayRef<int64_t> rhsBatchingDimensions, ::llvm::ArrayRef<int64_t> lhsContractingDimensions, ::llvm::ArrayRef<int64_t> rhsContractingDimensions) {
  return Base::get(context, std::move(lhsBatchingDimensions), std::move(rhsBatchingDimensions), std::move(lhsContractingDimensions), std::move(rhsContractingDimensions));
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getLhsBatchingDimensions() const {
  return getImpl()->lhsBatchingDimensions;
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getRhsBatchingDimensions() const {
  return getImpl()->rhsBatchingDimensions;
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getLhsContractingDimensions() const {
  return getImpl()->lhsContractingDimensions;
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getRhsContractingDimensions() const {
  return getImpl()->rhsContractingDimensions;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::DotDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct OutputOperandAliasAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, int64_t, ::llvm::ArrayRef<int64_t>>;
  OutputOperandAliasAttrStorage(::llvm::ArrayRef<int64_t> outputTupleIndices, int64_t operandIndex, ::llvm::ArrayRef<int64_t> operandTupleIndices) : outputTupleIndices(std::move(outputTupleIndices)), operandIndex(std::move(operandIndex)), operandTupleIndices(std::move(operandTupleIndices)) {}

  KeyTy getAsKey() const {
    return KeyTy(outputTupleIndices, operandIndex, operandTupleIndices);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (outputTupleIndices == std::get<0>(tblgenKey)) && (operandIndex == std::get<1>(tblgenKey)) && (operandTupleIndices == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static OutputOperandAliasAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto outputTupleIndices = std::move(std::get<0>(tblgenKey));
    auto operandIndex = std::move(std::get<1>(tblgenKey));
    auto operandTupleIndices = std::move(std::get<2>(tblgenKey));
    outputTupleIndices = allocator.copyInto(outputTupleIndices);
    operandTupleIndices = allocator.copyInto(operandTupleIndices);
    return new (allocator.allocate<OutputOperandAliasAttrStorage>()) OutputOperandAliasAttrStorage(std::move(outputTupleIndices), std::move(operandIndex), std::move(operandTupleIndices));
  }

  ::llvm::ArrayRef<int64_t> outputTupleIndices;
  int64_t operandIndex;
  ::llvm::ArrayRef<int64_t> operandTupleIndices;
};
} // namespace detail
OutputOperandAliasAttr OutputOperandAliasAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> outputTupleIndices, int64_t operandIndex, ::llvm::ArrayRef<int64_t> operandTupleIndices) {
  return Base::get(context, std::move(outputTupleIndices), std::move(operandIndex), std::move(operandTupleIndices));
}

::mlir::Attribute OutputOperandAliasAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_outputTupleIndices;
  ::mlir::FailureOr<int64_t> _result_operandIndex;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_operandTupleIndices;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'output_tuple_indices'
  if (odsParser.parseKeyword("output_tuple_indices")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'outputTupleIndices'
  _result_outputTupleIndices = parseDimSizes(odsParser);
  if (::mlir::failed(_result_outputTupleIndices)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_OutputOperandAlias parameter 'outputTupleIndices' which is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'operand_index'
  if (odsParser.parseKeyword("operand_index")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'operandIndex'
  _result_operandIndex = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_operandIndex)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_OutputOperandAlias parameter 'operandIndex' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'operand_tuple_indices'
  if (odsParser.parseKeyword("operand_tuple_indices")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'operandTupleIndices'
  _result_operandTupleIndices = parseDimSizes(odsParser);
  if (::mlir::failed(_result_operandTupleIndices)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_OutputOperandAlias parameter 'operandTupleIndices' which is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_outputTupleIndices));
  assert(::mlir::succeeded(_result_operandIndex));
  assert(::mlir::succeeded(_result_operandTupleIndices));
  return OutputOperandAliasAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_outputTupleIndices)),
      int64_t((*_result_operandIndex)),
      ::llvm::ArrayRef<int64_t>((*_result_operandTupleIndices)));
}

void OutputOperandAliasAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "output_tuple_indices";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  printDimSizes(odsPrinter, getOutputTupleIndices());
  odsPrinter << ",";
  odsPrinter << ' ' << "operand_index";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getOperandIndex());
  odsPrinter << ",";
  odsPrinter << ' ' << "operand_tuple_indices";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  printDimSizes(odsPrinter, getOperandTupleIndices());
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> OutputOperandAliasAttr::getOutputTupleIndices() const {
  return getImpl()->outputTupleIndices;
}

int64_t OutputOperandAliasAttr::getOperandIndex() const {
  return getImpl()->operandIndex;
}

::llvm::ArrayRef<int64_t> OutputOperandAliasAttr::getOperandTupleIndices() const {
  return getImpl()->operandTupleIndices;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::OutputOperandAliasAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ChannelHandleAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  ChannelHandleAttrStorage(int64_t handle, int64_t type) : handle(std::move(handle)), type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(handle, type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (handle == std::get<0>(tblgenKey)) && (type == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ChannelHandleAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto handle = std::move(std::get<0>(tblgenKey));
    auto type = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<ChannelHandleAttrStorage>()) ChannelHandleAttrStorage(std::move(handle), std::move(type));
  }

  int64_t handle;
  int64_t type;
};
} // namespace detail
ChannelHandleAttr ChannelHandleAttr::get(::mlir::MLIRContext *context, int64_t handle, int64_t type) {
  return Base::get(context, std::move(handle), std::move(type));
}

::mlir::Attribute ChannelHandleAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_handle;
  ::mlir::FailureOr<int64_t> _result_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_handle = false;
  bool _seen_type = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_handle && _paramKey == "handle") {
        _seen_handle = true;

        // Parse variable 'handle'
        _result_handle = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_handle)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ChannelHandle parameter 'handle' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_type && _paramKey == "type") {
        _seen_type = true;

        // Parse variable 'type'
        _result_type = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_type)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ChannelHandle parameter 'type' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_handle));
  assert(::mlir::succeeded(_result_type));
  return ChannelHandleAttr::get(odsParser.getContext(),
      int64_t((*_result_handle)),
      int64_t((*_result_type)));
}

void ChannelHandleAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "handle = ";
    odsPrinter.printStrippedAttrOrType(getHandle());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "type = ";
    odsPrinter.printStrippedAttrOrType(getType());
  }
  odsPrinter << ">";
}

int64_t ChannelHandleAttr::getHandle() const {
  return getImpl()->handle;
}

int64_t ChannelHandleAttr::getType() const {
  return getImpl()->type;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ChannelHandleAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct TypeExtensionsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>>;
  TypeExtensionsAttrStorage(::llvm::ArrayRef<int64_t> bounds) : bounds(std::move(bounds)) {}

  KeyTy getAsKey() const {
    return KeyTy(bounds);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (bounds == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeExtensionsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto bounds = std::move(std::get<0>(tblgenKey));
    bounds = allocator.copyInto(bounds);
    return new (allocator.allocate<TypeExtensionsAttrStorage>()) TypeExtensionsAttrStorage(std::move(bounds));
  }

  ::llvm::ArrayRef<int64_t> bounds;
};
} // namespace detail
TypeExtensionsAttr TypeExtensionsAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> bounds) {
  return Base::get(context, std::move(bounds));
}

::mlir::Attribute TypeExtensionsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_bounds;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'bounds'
  if (odsParser.parseKeyword("bounds")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseDimSizes(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_bounds));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_bounds)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'bounds'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_bounds));
  return TypeExtensionsAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_bounds)));
}

void TypeExtensionsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "bounds";
  odsPrinter << ' ' << "=";
  odsPrinter << " ";
  printDimSizes(odsPrinter,
    getBounds());
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> TypeExtensionsAttr::getBounds() const {
  return getImpl()->bounds;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TypeExtensionsAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ConvDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t, ::llvm::ArrayRef<int64_t>, int64_t, int64_t, ::llvm::ArrayRef<int64_t>, int64_t, int64_t, ::llvm::ArrayRef<int64_t>>;
  ConvDimensionNumbersAttrStorage(int64_t inputBatchDimension, int64_t inputFeatureDimension, ::llvm::ArrayRef<int64_t> inputSpatialDimensions, int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension, ::llvm::ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension, int64_t outputFeatureDimension, ::llvm::ArrayRef<int64_t> outputSpatialDimensions) : inputBatchDimension(std::move(inputBatchDimension)), inputFeatureDimension(std::move(inputFeatureDimension)), inputSpatialDimensions(std::move(inputSpatialDimensions)), kernelInputFeatureDimension(std::move(kernelInputFeatureDimension)), kernelOutputFeatureDimension(std::move(kernelOutputFeatureDimension)), kernelSpatialDimensions(std::move(kernelSpatialDimensions)), outputBatchDimension(std::move(outputBatchDimension)), outputFeatureDimension(std::move(outputFeatureDimension)), outputSpatialDimensions(std::move(outputSpatialDimensions)) {}

  KeyTy getAsKey() const {
    return KeyTy(inputBatchDimension, inputFeatureDimension, inputSpatialDimensions, kernelInputFeatureDimension, kernelOutputFeatureDimension, kernelSpatialDimensions, outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (inputBatchDimension == std::get<0>(tblgenKey)) && (inputFeatureDimension == std::get<1>(tblgenKey)) && (inputSpatialDimensions == std::get<2>(tblgenKey)) && (kernelInputFeatureDimension == std::get<3>(tblgenKey)) && (kernelOutputFeatureDimension == std::get<4>(tblgenKey)) && (kernelSpatialDimensions == std::get<5>(tblgenKey)) && (outputBatchDimension == std::get<6>(tblgenKey)) && (outputFeatureDimension == std::get<7>(tblgenKey)) && (outputSpatialDimensions == std::get<8>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey), std::get<7>(tblgenKey), std::get<8>(tblgenKey));
  }

  static ConvDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto inputBatchDimension = std::move(std::get<0>(tblgenKey));
    auto inputFeatureDimension = std::move(std::get<1>(tblgenKey));
    auto inputSpatialDimensions = std::move(std::get<2>(tblgenKey));
    auto kernelInputFeatureDimension = std::move(std::get<3>(tblgenKey));
    auto kernelOutputFeatureDimension = std::move(std::get<4>(tblgenKey));
    auto kernelSpatialDimensions = std::move(std::get<5>(tblgenKey));
    auto outputBatchDimension = std::move(std::get<6>(tblgenKey));
    auto outputFeatureDimension = std::move(std::get<7>(tblgenKey));
    auto outputSpatialDimensions = std::move(std::get<8>(tblgenKey));
    inputSpatialDimensions = allocator.copyInto(inputSpatialDimensions);
    kernelSpatialDimensions = allocator.copyInto(kernelSpatialDimensions);
    outputSpatialDimensions = allocator.copyInto(outputSpatialDimensions);
    return new (allocator.allocate<ConvDimensionNumbersAttrStorage>()) ConvDimensionNumbersAttrStorage(std::move(inputBatchDimension), std::move(inputFeatureDimension), std::move(inputSpatialDimensions), std::move(kernelInputFeatureDimension), std::move(kernelOutputFeatureDimension), std::move(kernelSpatialDimensions), std::move(outputBatchDimension), std::move(outputFeatureDimension), std::move(outputSpatialDimensions));
  }

  int64_t inputBatchDimension;
  int64_t inputFeatureDimension;
  ::llvm::ArrayRef<int64_t> inputSpatialDimensions;
  int64_t kernelInputFeatureDimension;
  int64_t kernelOutputFeatureDimension;
  ::llvm::ArrayRef<int64_t> kernelSpatialDimensions;
  int64_t outputBatchDimension;
  int64_t outputFeatureDimension;
  ::llvm::ArrayRef<int64_t> outputSpatialDimensions;
};
} // namespace detail
ConvDimensionNumbersAttr ConvDimensionNumbersAttr::get(::mlir::MLIRContext *context, int64_t inputBatchDimension, int64_t inputFeatureDimension, ::llvm::ArrayRef<int64_t> inputSpatialDimensions, int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension, ::llvm::ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension, int64_t outputFeatureDimension, ::llvm::ArrayRef<int64_t> outputSpatialDimensions) {
  return Base::get(context, std::move(inputBatchDimension), std::move(inputFeatureDimension), std::move(inputSpatialDimensions), std::move(kernelInputFeatureDimension), std::move(kernelOutputFeatureDimension), std::move(kernelSpatialDimensions), std::move(outputBatchDimension), std::move(outputFeatureDimension), std::move(outputSpatialDimensions));
}

int64_t ConvDimensionNumbersAttr::getInputBatchDimension() const {
  return getImpl()->inputBatchDimension;
}

int64_t ConvDimensionNumbersAttr::getInputFeatureDimension() const {
  return getImpl()->inputFeatureDimension;
}

::llvm::ArrayRef<int64_t> ConvDimensionNumbersAttr::getInputSpatialDimensions() const {
  return getImpl()->inputSpatialDimensions;
}

int64_t ConvDimensionNumbersAttr::getKernelInputFeatureDimension() const {
  return getImpl()->kernelInputFeatureDimension;
}

int64_t ConvDimensionNumbersAttr::getKernelOutputFeatureDimension() const {
  return getImpl()->kernelOutputFeatureDimension;
}

::llvm::ArrayRef<int64_t> ConvDimensionNumbersAttr::getKernelSpatialDimensions() const {
  return getImpl()->kernelSpatialDimensions;
}

int64_t ConvDimensionNumbersAttr::getOutputBatchDimension() const {
  return getImpl()->outputBatchDimension;
}

int64_t ConvDimensionNumbersAttr::getOutputFeatureDimension() const {
  return getImpl()->outputFeatureDimension;
}

::llvm::ArrayRef<int64_t> ConvDimensionNumbersAttr::getOutputSpatialDimensions() const {
  return getImpl()->outputSpatialDimensions;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ConvDimensionNumbersAttr)

#endif  // GET_ATTRDEF_CLASSES

