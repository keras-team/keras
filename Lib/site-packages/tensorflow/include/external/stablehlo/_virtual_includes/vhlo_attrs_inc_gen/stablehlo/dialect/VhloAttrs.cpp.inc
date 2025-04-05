/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::vhlo::ArrayV1Attr,
::mlir::vhlo::BooleanV1Attr,
::mlir::vhlo::DictionaryV1Attr,
::mlir::vhlo::FloatV1Attr,
::mlir::vhlo::IntegerV1Attr,
::mlir::vhlo::OutputOperandAliasV1Attr,
::mlir::vhlo::StringV1Attr,
::mlir::vhlo::TensorV1Attr,
::mlir::vhlo::TypeV1Attr,
::mlir::vhlo::TypeExtensionsV1Attr,
::mlir::vhlo::ComparisonDirectionV1Attr,
::mlir::vhlo::ComparisonTypeV1Attr,
::mlir::vhlo::CustomCallApiVersionV1Attr,
::mlir::vhlo::FftTypeV1Attr,
::mlir::vhlo::PrecisionV1Attr,
::mlir::vhlo::RngAlgorithmV1Attr,
::mlir::vhlo::RngDistributionV1Attr,
::mlir::vhlo::TransposeV1Attr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::vhlo::ArrayV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::ArrayV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::BooleanV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::BooleanV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::DictionaryV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::DictionaryV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::OutputOperandAliasV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::OutputOperandAliasV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::StringV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::StringV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::TensorV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::TensorV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::TypeV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::TypeV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::TypeExtensionsV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::TypeExtensionsV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::ComparisonDirectionV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::ComparisonDirectionV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::ComparisonTypeV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::ComparisonTypeV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::CustomCallApiVersionV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::CustomCallApiVersionV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FftTypeV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FftTypeV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::PrecisionV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::PrecisionV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::RngAlgorithmV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::RngAlgorithmV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::RngDistributionV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::RngDistributionV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::TransposeV1Attr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::TransposeV1Attr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::vhlo::ArrayV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::ArrayV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::BooleanV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::BooleanV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::DictionaryV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::DictionaryV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::FloatV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::IntegerV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::OutputOperandAliasV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::OutputOperandAliasV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::StringV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::StringV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::TensorV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::TensorV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::TypeV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::TypeV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::TypeExtensionsV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::TypeExtensionsV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::ComparisonDirectionV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::ComparisonDirectionV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::ComparisonTypeV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::ComparisonTypeV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::CustomCallApiVersionV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::CustomCallApiVersionV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FftTypeV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::FftTypeV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::PrecisionV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::PrecisionV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::RngAlgorithmV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::RngAlgorithmV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::RngDistributionV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::RngDistributionV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::TransposeV1Attr>([&](auto t) {
      printer << ::mlir::vhlo::TransposeV1Attr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace vhlo {
namespace detail {
struct ArrayV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<mlir::Attribute>>;
  ArrayV1AttrStorage(::llvm::ArrayRef<mlir::Attribute> value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ArrayV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<ArrayV1AttrStorage>()) ArrayV1AttrStorage(std::move(value));
  }

  ::llvm::ArrayRef<mlir::Attribute> value;
};
} // namespace detail
LogicalResult ArrayV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, ArrayRef<mlir::Attribute> value) {
  if (!allFromVhlo(value)) return errFn() << "expected array of VHLO attriutes";
  return success();
}
ArrayV1Attr ArrayV1Attr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<mlir::Attribute> value) {
  return Base::get(context, std::move(value));
}

ArrayV1Attr ArrayV1Attr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<mlir::Attribute> value) {
  return Base::getChecked(emitError, context, value);
}

::llvm::LogicalResult ArrayV1Attr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<mlir::Attribute> value) {
  if (::mlir::failed(verify(emitError, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute ArrayV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<mlir::Attribute>> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseAttributeArray(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_value));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_value)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'value'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return odsParser.getChecked<ArrayV1Attr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<mlir::Attribute>((*_result_value)));
}

void ArrayV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printAttributeArray(odsPrinter,
    getValue());
  odsPrinter << ">";
}

::llvm::ArrayRef<mlir::Attribute> ArrayV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::ArrayV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct BooleanV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<bool>;
  BooleanV1AttrStorage(bool value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static BooleanV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<BooleanV1AttrStorage>()) BooleanV1AttrStorage(std::move(value));
  }

  bool value;
};
} // namespace detail
BooleanV1Attr BooleanV1Attr::get(::mlir::MLIRContext *context, bool value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute BooleanV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<bool> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<bool>::parse(odsParser);
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_BooleanAttrV1 parameter 'value' which is to be a `bool`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return BooleanV1Attr::get(odsParser.getContext(),
      bool((*_result_value)));
}

void BooleanV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getValue());
  odsPrinter << ">";
}

bool BooleanV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::BooleanV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct DictionaryV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>>;
  DictionaryV1AttrStorage(::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DictionaryV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<DictionaryV1AttrStorage>()) DictionaryV1AttrStorage(std::move(value));
  }

  ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value;
};
} // namespace detail
LogicalResult DictionaryV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
    ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value) {
  for (auto & entry : value)
    if (!isFromVhlo(entry.first) || !isFromVhlo(entry.second))
      errFn() << "expected VHLO attribute";
  return success();
}
DictionaryV1Attr DictionaryV1Attr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value) {
  return Base::get(context, std::move(value));
}

DictionaryV1Attr DictionaryV1Attr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value) {
  return Base::getChecked(emitError, context, value);
}

::llvm::LogicalResult DictionaryV1Attr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value) {
  if (::mlir::failed(verify(emitError, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute DictionaryV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>>> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseAttributeDictionary(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_value));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_value)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'value'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return odsParser.getChecked<DictionaryV1Attr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>((*_result_value)));
}

void DictionaryV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printAttributeDictionary(odsPrinter,
    getValue());
  odsPrinter << ">";
}

::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> DictionaryV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::DictionaryV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct FloatV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<mlir::Type, ::llvm::APFloat>;
  FloatV1AttrStorage(mlir::Type type, ::llvm::APFloat value) : type(std::move(type)), value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (value.bitwiseIsEqual(std::get<1>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FloatV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto value = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<FloatV1AttrStorage>()) FloatV1AttrStorage(std::move(type), std::move(value));
  }

  mlir::Type type;
  ::llvm::APFloat value;
};
} // namespace detail
LogicalResult FloatV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type type, APFloat value) {
  if (!isFromVhlo(type)) return errFn() << "expected VHLO type";
  return success();
}
FloatV1Attr FloatV1Attr::get(::mlir::MLIRContext *context, mlir::Type type, ::llvm::APFloat value) {
  return Base::get(context, std::move(type), std::move(value));
}

FloatV1Attr FloatV1Attr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, mlir::Type type, ::llvm::APFloat value) {
  return Base::getChecked(emitError, context, type, value);
}

::llvm::LogicalResult FloatV1Attr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, ::llvm::APFloat value) {
  if (::mlir::failed(verify(emitError, type, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute FloatV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<mlir::Type> _result_type;
  ::mlir::FailureOr<::llvm::APFloat> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = 
      [&]() -> FailureOr<llvm::APFloat> {
        double value;
        if (failed(odsParser.parseFloat(value))) {
          return failure();
        }
        return APFloat(value);
      }()
    ;
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_FloatAttrV1 parameter 'value' which is to be a `::llvm::APFloat`");
    return {};
  }
  // Parse literal ':'
  if (odsParser.parseColon()) return {};

  // Parse variable 'type'
  _result_type = ::mlir::FieldParser<mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_FloatAttrV1 parameter 'type' which is to be a `mlir::Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_type));
  assert(::mlir::succeeded(_result_value));
  return odsParser.getChecked<FloatV1Attr>(odsLoc, odsParser.getContext(),
      mlir::Type((*_result_type)),
      ::llvm::APFloat((*_result_value)));
}

void FloatV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printFloat(getValue());;
  odsPrinter << ' ' << ":";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getType());
  odsPrinter << ">";
}

mlir::Type FloatV1Attr::getType() const {
  return getImpl()->type;
}

::llvm::APFloat FloatV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct IntegerV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<mlir::Type, APInt>;
  IntegerV1AttrStorage(mlir::Type type, APInt value) : type(std::move(type)), value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (value == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static IntegerV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto value = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<IntegerV1AttrStorage>()) IntegerV1AttrStorage(std::move(type), std::move(value));
  }

  mlir::Type type;
  APInt value;
};
} // namespace detail
LogicalResult IntegerV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type type, APInt value) {
  if (!isFromVhlo(type)) return errFn() << "expected VHLO type";
  return success();
}
IntegerV1Attr IntegerV1Attr::get(::mlir::MLIRContext *context, mlir::Type type, APInt value) {
  return Base::get(context, std::move(type), std::move(value));
}

IntegerV1Attr IntegerV1Attr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, mlir::Type type, APInt value) {
  return Base::getChecked(emitError, context, type, value);
}

::llvm::LogicalResult IntegerV1Attr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, APInt value) {
  if (::mlir::failed(verify(emitError, type, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

mlir::Type IntegerV1Attr::getType() const {
  return getImpl()->type;
}

APInt IntegerV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct OutputOperandAliasV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, int64_t, ::llvm::ArrayRef<int64_t>>;
  OutputOperandAliasV1AttrStorage(::llvm::ArrayRef<int64_t> outputTupleIndices, int64_t operandIndex, ::llvm::ArrayRef<int64_t> operandTupleIndices) : outputTupleIndices(std::move(outputTupleIndices)), operandIndex(std::move(operandIndex)), operandTupleIndices(std::move(operandTupleIndices)) {}

  KeyTy getAsKey() const {
    return KeyTy(outputTupleIndices, operandIndex, operandTupleIndices);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (outputTupleIndices == std::get<0>(tblgenKey)) && (operandIndex == std::get<1>(tblgenKey)) && (operandTupleIndices == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static OutputOperandAliasV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto outputTupleIndices = std::move(std::get<0>(tblgenKey));
    auto operandIndex = std::move(std::get<1>(tblgenKey));
    auto operandTupleIndices = std::move(std::get<2>(tblgenKey));
    outputTupleIndices = allocator.copyInto(outputTupleIndices);
    operandTupleIndices = allocator.copyInto(operandTupleIndices);
    return new (allocator.allocate<OutputOperandAliasV1AttrStorage>()) OutputOperandAliasV1AttrStorage(std::move(outputTupleIndices), std::move(operandIndex), std::move(operandTupleIndices));
  }

  ::llvm::ArrayRef<int64_t> outputTupleIndices;
  int64_t operandIndex;
  ::llvm::ArrayRef<int64_t> operandTupleIndices;
};
} // namespace detail
OutputOperandAliasV1Attr OutputOperandAliasV1Attr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> outputTupleIndices, int64_t operandIndex, ::llvm::ArrayRef<int64_t> operandTupleIndices) {
  return Base::get(context, std::move(outputTupleIndices), std::move(operandIndex), std::move(operandTupleIndices));
}

::mlir::Attribute OutputOperandAliasV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_outputTupleIndices;
  ::mlir::FailureOr<int64_t> _result_operandIndex;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_operandTupleIndices;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_outputTupleIndices = false;
  bool _seen_operandIndex = false;
  bool _seen_operandTupleIndices = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_outputTupleIndices && _paramKey == "outputTupleIndices") {
        _seen_outputTupleIndices = true;

        // Parse variable 'outputTupleIndices'
        _result_outputTupleIndices = mlir::hlo::parseDimSizes(odsParser);
        if (::mlir::failed(_result_outputTupleIndices)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_OutputOperandAliasAttrV1 parameter 'outputTupleIndices' which is to be a `::llvm::ArrayRef<int64_t>`");
          return {};
        }
      } else if (!_seen_operandIndex && _paramKey == "operandIndex") {
        _seen_operandIndex = true;

        // Parse variable 'operandIndex'
        _result_operandIndex = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_operandIndex)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_OutputOperandAliasAttrV1 parameter 'operandIndex' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_operandTupleIndices && _paramKey == "operandTupleIndices") {
        _seen_operandTupleIndices = true;

        // Parse variable 'operandTupleIndices'
        _result_operandTupleIndices = mlir::hlo::parseDimSizes(odsParser);
        if (::mlir::failed(_result_operandTupleIndices)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_OutputOperandAliasAttrV1 parameter 'operandTupleIndices' which is to be a `::llvm::ArrayRef<int64_t>`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 3; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 3 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_outputTupleIndices));
  assert(::mlir::succeeded(_result_operandIndex));
  assert(::mlir::succeeded(_result_operandTupleIndices));
  return OutputOperandAliasV1Attr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_outputTupleIndices)),
      int64_t((*_result_operandIndex)),
      ::llvm::ArrayRef<int64_t>((*_result_operandTupleIndices)));
}

void OutputOperandAliasV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "outputTupleIndices = ";
    mlir::hlo::printDimSizes(odsPrinter, getOutputTupleIndices());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "operandIndex = ";
    odsPrinter.printStrippedAttrOrType(getOperandIndex());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "operandTupleIndices = ";
    mlir::hlo::printDimSizes(odsPrinter, getOperandTupleIndices());
  }
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> OutputOperandAliasV1Attr::getOutputTupleIndices() const {
  return getImpl()->outputTupleIndices;
}

int64_t OutputOperandAliasV1Attr::getOperandIndex() const {
  return getImpl()->operandIndex;
}

::llvm::ArrayRef<int64_t> OutputOperandAliasV1Attr::getOperandTupleIndices() const {
  return getImpl()->operandTupleIndices;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::OutputOperandAliasV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct StringV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  StringV1AttrStorage(::llvm::StringRef value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static StringV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<StringV1AttrStorage>()) StringV1AttrStorage(std::move(value));
  }

  ::llvm::StringRef value;
};
} // namespace detail
StringV1Attr StringV1Attr::get(::mlir::MLIRContext *context, ::llvm::StringRef value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute StringV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::string> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseEscapedString(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_value));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_value)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'value'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return StringV1Attr::get(odsParser.getContext(),
      ::llvm::StringRef((*_result_value)));
}

void StringV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printEscapedString(odsPrinter,
    getValue());
  odsPrinter << ">";
}

::llvm::StringRef StringV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::StringV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct TensorV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type, ::llvm::ArrayRef<char>>;
  TensorV1AttrStorage(::mlir::Type type, ::llvm::ArrayRef<char> data) : type(std::move(type)), data(std::move(data)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, data);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (data == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TensorV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto data = std::move(std::get<1>(tblgenKey));
    data = allocator.copyInto(data);
    return new (allocator.allocate<TensorV1AttrStorage>()) TensorV1AttrStorage(std::move(type), std::move(data));
  }

  ::mlir::Type type;
  ::llvm::ArrayRef<char> data;
};
} // namespace detail
LogicalResult TensorV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type type, ArrayRef<char>) {
  if (!isFromVhlo(type)) errFn() << "expected VHLO type";
  return success();
}
TensorV1Attr TensorV1Attr::get(::mlir::MLIRContext *context, ::mlir::Type type, ::llvm::ArrayRef<char> data) {
  return Base::get(context, std::move(type), std::move(data));
}

TensorV1Attr TensorV1Attr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type type, ::llvm::ArrayRef<char> data) {
  return Base::getChecked(emitError, context, type, data);
}

::llvm::LogicalResult TensorV1Attr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, ::llvm::ArrayRef<char> data) {
  if (::mlir::failed(verify(emitError, type, data)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type TensorV1Attr::getType() const {
  return getImpl()->type;
}

::llvm::ArrayRef<char> TensorV1Attr::getData() const {
  return getImpl()->data;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::TensorV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct TypeV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type>;
  TypeV1AttrStorage(::mlir::Type value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TypeV1AttrStorage>()) TypeV1AttrStorage(std::move(value));
  }

  ::mlir::Type value;
};
} // namespace detail
LogicalResult TypeV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type value) {
  if (!isFromVhlo(value)) return errFn() << "expected VHLO type";
  return success();
}
TypeV1Attr TypeV1Attr::get(::mlir::MLIRContext *context, ::mlir::Type value) {
  return Base::get(context, std::move(value));
}

TypeV1Attr TypeV1Attr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type value) {
  return Base::getChecked(emitError, context, value);
}

::llvm::LogicalResult TypeV1Attr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type value) {
  if (::mlir::failed(verify(emitError, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute TypeV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::Type> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_TypeAttrV1 parameter 'value' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return odsParser.getChecked<TypeV1Attr>(odsLoc, odsParser.getContext(),
      ::mlir::Type((*_result_value)));
}

void TypeV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getValue());
  odsPrinter << ">";
}

::mlir::Type TypeV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::TypeV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct TypeExtensionsV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>>;
  TypeExtensionsV1AttrStorage(::llvm::ArrayRef<int64_t> bounds) : bounds(std::move(bounds)) {}

  KeyTy getAsKey() const {
    return KeyTy(bounds);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (bounds == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeExtensionsV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto bounds = std::move(std::get<0>(tblgenKey));
    bounds = allocator.copyInto(bounds);
    return new (allocator.allocate<TypeExtensionsV1AttrStorage>()) TypeExtensionsV1AttrStorage(std::move(bounds));
  }

  ::llvm::ArrayRef<int64_t> bounds;
};
} // namespace detail
TypeExtensionsV1Attr TypeExtensionsV1Attr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> bounds) {
  return Base::get(context, std::move(bounds));
}

::mlir::Attribute TypeExtensionsV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_bounds;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_bounds = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_bounds && _paramKey == "bounds") {
        _seen_bounds = true;

        // Parse variable 'bounds'
        _result_bounds = mlir::hlo::parseDimSizes(odsParser);
        if (::mlir::failed(_result_bounds)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_TypeExtensionsAttrV1 parameter 'bounds' which is to be a `::llvm::ArrayRef<int64_t>`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_bounds));
  return TypeExtensionsV1Attr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_bounds)));
}

void TypeExtensionsV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "bounds = ";
    mlir::hlo::printDimSizes(odsPrinter, getBounds());
  }
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> TypeExtensionsV1Attr::getBounds() const {
  return getImpl()->bounds;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::TypeExtensionsV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct ComparisonDirectionV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::ComparisonDirectionV1>;
  ComparisonDirectionV1AttrStorage(::mlir::vhlo::ComparisonDirectionV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonDirectionV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComparisonDirectionV1AttrStorage>()) ComparisonDirectionV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::ComparisonDirectionV1 value;
};
} // namespace detail
ComparisonDirectionV1Attr ComparisonDirectionV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::ComparisonDirectionV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ComparisonDirectionV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::ComparisonDirectionV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::ComparisonDirectionV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeComparisonDirectionV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::ComparisonDirectionV1" << " to be one of: " << "EQ" << ", " << "NE" << ", " << "GE" << ", " << "GT" << ", " << "LE" << ", " << "LT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_ComparisonDirectionAttrV1 parameter 'value' which is to be a `::mlir::vhlo::ComparisonDirectionV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonDirectionV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::ComparisonDirectionV1((*_result_value)));
}

void ComparisonDirectionV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonDirectionV1(getValue());
}

::mlir::vhlo::ComparisonDirectionV1 ComparisonDirectionV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::ComparisonDirectionV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct ComparisonTypeV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::ComparisonTypeV1>;
  ComparisonTypeV1AttrStorage(::mlir::vhlo::ComparisonTypeV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonTypeV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComparisonTypeV1AttrStorage>()) ComparisonTypeV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::ComparisonTypeV1 value;
};
} // namespace detail
ComparisonTypeV1Attr ComparisonTypeV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::ComparisonTypeV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ComparisonTypeV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::ComparisonTypeV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::ComparisonTypeV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeComparisonTypeV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::ComparisonTypeV1" << " to be one of: " << "NOTYPE" << ", " << "FLOAT" << ", " << "TOTALORDER" << ", " << "SIGNED" << ", " << "UNSIGNED")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_ComparisonTypeAttrV1 parameter 'value' which is to be a `::mlir::vhlo::ComparisonTypeV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonTypeV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::ComparisonTypeV1((*_result_value)));
}

void ComparisonTypeV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonTypeV1(getValue());
}

::mlir::vhlo::ComparisonTypeV1 ComparisonTypeV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::ComparisonTypeV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct CustomCallApiVersionV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::CustomCallApiVersionV1>;
  CustomCallApiVersionV1AttrStorage(::mlir::vhlo::CustomCallApiVersionV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CustomCallApiVersionV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CustomCallApiVersionV1AttrStorage>()) CustomCallApiVersionV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::CustomCallApiVersionV1 value;
};
} // namespace detail
CustomCallApiVersionV1Attr CustomCallApiVersionV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::CustomCallApiVersionV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CustomCallApiVersionV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::CustomCallApiVersionV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::CustomCallApiVersionV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeCustomCallApiVersionV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::CustomCallApiVersionV1" << " to be one of: " << "API_VERSION_UNSPECIFIED" << ", " << "API_VERSION_ORIGINAL" << ", " << "API_VERSION_STATUS_RETURNING" << ", " << "API_VERSION_STATUS_RETURNING_UNIFIED" << ", " << "API_VERSION_TYPED_FFI")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_CustomCallApiVersionAttrV1 parameter 'value' which is to be a `::mlir::vhlo::CustomCallApiVersionV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return CustomCallApiVersionV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::CustomCallApiVersionV1((*_result_value)));
}

void CustomCallApiVersionV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyCustomCallApiVersionV1(getValue());
}

::mlir::vhlo::CustomCallApiVersionV1 CustomCallApiVersionV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::CustomCallApiVersionV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct FftTypeV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::FftTypeV1>;
  FftTypeV1AttrStorage(::mlir::vhlo::FftTypeV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FftTypeV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<FftTypeV1AttrStorage>()) FftTypeV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::FftTypeV1 value;
};
} // namespace detail
FftTypeV1Attr FftTypeV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::FftTypeV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute FftTypeV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::FftTypeV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::FftTypeV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeFftTypeV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::FftTypeV1" << " to be one of: " << "FFT" << ", " << "IFFT" << ", " << "RFFT" << ", " << "IRFFT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_FftTypeAttrV1 parameter 'value' which is to be a `::mlir::vhlo::FftTypeV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return FftTypeV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::FftTypeV1((*_result_value)));
}

void FftTypeV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyFftTypeV1(getValue());
}

::mlir::vhlo::FftTypeV1 FftTypeV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FftTypeV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct PrecisionV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::PrecisionV1>;
  PrecisionV1AttrStorage(::mlir::vhlo::PrecisionV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PrecisionV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PrecisionV1AttrStorage>()) PrecisionV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::PrecisionV1 value;
};
} // namespace detail
PrecisionV1Attr PrecisionV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::PrecisionV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute PrecisionV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::PrecisionV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::PrecisionV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizePrecisionV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::PrecisionV1" << " to be one of: " << "DEFAULT" << ", " << "HIGH" << ", " << "HIGHEST")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_PrecisionAttrV1 parameter 'value' which is to be a `::mlir::vhlo::PrecisionV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return PrecisionV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::PrecisionV1((*_result_value)));
}

void PrecisionV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyPrecisionV1(getValue());
}

::mlir::vhlo::PrecisionV1 PrecisionV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::PrecisionV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct RngAlgorithmV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::RngAlgorithmV1>;
  RngAlgorithmV1AttrStorage(::mlir::vhlo::RngAlgorithmV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RngAlgorithmV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RngAlgorithmV1AttrStorage>()) RngAlgorithmV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::RngAlgorithmV1 value;
};
} // namespace detail
RngAlgorithmV1Attr RngAlgorithmV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::RngAlgorithmV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RngAlgorithmV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::RngAlgorithmV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::RngAlgorithmV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeRngAlgorithmV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::RngAlgorithmV1" << " to be one of: " << "DEFAULT" << ", " << "THREE_FRY" << ", " << "PHILOX")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_RngAlgorithmAttrV1 parameter 'value' which is to be a `::mlir::vhlo::RngAlgorithmV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RngAlgorithmV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::RngAlgorithmV1((*_result_value)));
}

void RngAlgorithmV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRngAlgorithmV1(getValue());
}

::mlir::vhlo::RngAlgorithmV1 RngAlgorithmV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::RngAlgorithmV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct RngDistributionV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::RngDistributionV1>;
  RngDistributionV1AttrStorage(::mlir::vhlo::RngDistributionV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RngDistributionV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RngDistributionV1AttrStorage>()) RngDistributionV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::RngDistributionV1 value;
};
} // namespace detail
RngDistributionV1Attr RngDistributionV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::RngDistributionV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RngDistributionV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::RngDistributionV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::RngDistributionV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeRngDistributionV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::RngDistributionV1" << " to be one of: " << "UNIFORM" << ", " << "NORMAL")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_RngDistributionAttrV1 parameter 'value' which is to be a `::mlir::vhlo::RngDistributionV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RngDistributionV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::RngDistributionV1((*_result_value)));
}

void RngDistributionV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRngDistributionV1(getValue());
}

::mlir::vhlo::RngDistributionV1 RngDistributionV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::RngDistributionV1Attr)
namespace mlir {
namespace vhlo {
namespace detail {
struct TransposeV1AttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vhlo::TransposeV1>;
  TransposeV1AttrStorage(::mlir::vhlo::TransposeV1 value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TransposeV1AttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TransposeV1AttrStorage>()) TransposeV1AttrStorage(std::move(value));
  }

  ::mlir::vhlo::TransposeV1 value;
};
} // namespace detail
TransposeV1Attr TransposeV1Attr::get(::mlir::MLIRContext *context, ::mlir::vhlo::TransposeV1 value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TransposeV1Attr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vhlo::TransposeV1> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vhlo::TransposeV1> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vhlo::symbolizeTransposeV1(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vhlo::TransposeV1" << " to be one of: " << "TRANSPOSE_INVALID" << ", " << "NO_TRANSPOSE" << ", " << "TRANSPOSE" << ", " << "ADJOINT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_TransposeAttrV1 parameter 'value' which is to be a `::mlir::vhlo::TransposeV1`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TransposeV1Attr::get(odsParser.getContext(),
      ::mlir::vhlo::TransposeV1((*_result_value)));
}

void TransposeV1Attr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTransposeV1(getValue());
}

::mlir::vhlo::TransposeV1 TransposeV1Attr::getValue() const {
  return getImpl()->value;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::TransposeV1Attr)

#endif  // GET_ATTRDEF_CLASSES

