/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::omp::ClauseCancellationConstructTypeAttr,
::mlir::omp::ClauseGrainsizeTypeAttr,
::mlir::omp::ClauseMemoryOrderKindAttr,
::mlir::omp::ClauseNumTasksTypeAttr,
::mlir::omp::ClauseOrderKindAttr,
::mlir::omp::ClauseProcBindKindAttr,
::mlir::omp::ClauseScheduleKindAttr,
::mlir::omp::DeclareTargetCaptureClauseAttr,
::mlir::omp::ClauseDependAttr,
::mlir::omp::ClauseRequiresAttr,
::mlir::omp::ClauseTaskDependAttr,
::mlir::omp::DataSharingClauseTypeAttr,
::mlir::omp::DeclareTargetDeviceTypeAttr,
::mlir::omp::OrderModifierAttr,
::mlir::omp::ScheduleModifierAttr,
::mlir::omp::VariableCaptureKindAttr,
::mlir::omp::DeclareTargetAttr,
::mlir::omp::FlagsAttr,
::mlir::omp::VersionAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::omp::ClauseCancellationConstructTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseCancellationConstructTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseGrainsizeTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseGrainsizeTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseMemoryOrderKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseMemoryOrderKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseNumTasksTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseNumTasksTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseOrderKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseOrderKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseProcBindKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseProcBindKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseScheduleKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseScheduleKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::DeclareTargetCaptureClauseAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::DeclareTargetCaptureClauseAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseDependAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseDependAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseRequiresAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseRequiresAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ClauseTaskDependAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ClauseTaskDependAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::DataSharingClauseTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::DataSharingClauseTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::DeclareTargetDeviceTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::DeclareTargetDeviceTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::OrderModifierAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::OrderModifierAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::ScheduleModifierAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::ScheduleModifierAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::VariableCaptureKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::VariableCaptureKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::DeclareTargetAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::DeclareTargetAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::FlagsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::FlagsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::omp::VersionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::omp::VersionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::omp::ClauseCancellationConstructTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseCancellationConstructTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseGrainsizeTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseGrainsizeTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseMemoryOrderKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseMemoryOrderKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseNumTasksTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseNumTasksTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseOrderKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseOrderKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseProcBindKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseProcBindKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseScheduleKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseScheduleKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::DeclareTargetCaptureClauseAttr>([&](auto t) {
      printer << ::mlir::omp::DeclareTargetCaptureClauseAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseDependAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseDependAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseRequiresAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseRequiresAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseTaskDependAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseTaskDependAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::DataSharingClauseTypeAttr>([&](auto t) {
      printer << ::mlir::omp::DataSharingClauseTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::DeclareTargetDeviceTypeAttr>([&](auto t) {
      printer << ::mlir::omp::DeclareTargetDeviceTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::OrderModifierAttr>([&](auto t) {
      printer << ::mlir::omp::OrderModifierAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ScheduleModifierAttr>([&](auto t) {
      printer << ::mlir::omp::ScheduleModifierAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::VariableCaptureKindAttr>([&](auto t) {
      printer << ::mlir::omp::VariableCaptureKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::DeclareTargetAttr>([&](auto t) {
      printer << ::mlir::omp::DeclareTargetAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::FlagsAttr>([&](auto t) {
      printer << ::mlir::omp::FlagsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::VersionAttr>([&](auto t) {
      printer << ::mlir::omp::VersionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace omp {
namespace detail {
struct ClauseCancellationConstructTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseCancellationConstructType>;
  ClauseCancellationConstructTypeAttrStorage(::mlir::omp::ClauseCancellationConstructType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseCancellationConstructTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseCancellationConstructTypeAttrStorage>()) ClauseCancellationConstructTypeAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseCancellationConstructType value;
};
} // namespace detail
ClauseCancellationConstructTypeAttr ClauseCancellationConstructTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseCancellationConstructType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseCancellationConstructTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseCancellationConstructType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseCancellationConstructType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseCancellationConstructType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseCancellationConstructType" << " to be one of: " << "parallel" << ", " << "loop" << ", " << "sections" << ", " << "taskgroup")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CancellationConstructTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseCancellationConstructType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseCancellationConstructTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseCancellationConstructType((*_result_value)));
}

void ClauseCancellationConstructTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseCancellationConstructType(getValue());
}

::mlir::omp::ClauseCancellationConstructType ClauseCancellationConstructTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseCancellationConstructTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseGrainsizeTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseGrainsizeType>;
  ClauseGrainsizeTypeAttrStorage(::mlir::omp::ClauseGrainsizeType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseGrainsizeTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseGrainsizeTypeAttrStorage>()) ClauseGrainsizeTypeAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseGrainsizeType value;
};
} // namespace detail
ClauseGrainsizeTypeAttr ClauseGrainsizeTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseGrainsizeType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseGrainsizeTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseGrainsizeType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseGrainsizeType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseGrainsizeType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseGrainsizeType" << " to be one of: " << "strict")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GrainsizeTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseGrainsizeType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseGrainsizeTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseGrainsizeType((*_result_value)));
}

void ClauseGrainsizeTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseGrainsizeType(getValue());
}

::mlir::omp::ClauseGrainsizeType ClauseGrainsizeTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseGrainsizeTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseMemoryOrderKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseMemoryOrderKind>;
  ClauseMemoryOrderKindAttrStorage(::mlir::omp::ClauseMemoryOrderKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseMemoryOrderKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseMemoryOrderKindAttrStorage>()) ClauseMemoryOrderKindAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseMemoryOrderKind value;
};
} // namespace detail
ClauseMemoryOrderKindAttr ClauseMemoryOrderKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseMemoryOrderKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseMemoryOrderKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseMemoryOrderKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseMemoryOrderKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseMemoryOrderKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseMemoryOrderKind" << " to be one of: " << "seq_cst" << ", " << "acq_rel" << ", " << "acquire" << ", " << "release" << ", " << "relaxed")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MemoryOrderKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseMemoryOrderKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseMemoryOrderKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseMemoryOrderKind((*_result_value)));
}

void ClauseMemoryOrderKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseMemoryOrderKind(getValue());
}

::mlir::omp::ClauseMemoryOrderKind ClauseMemoryOrderKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseMemoryOrderKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseNumTasksTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseNumTasksType>;
  ClauseNumTasksTypeAttrStorage(::mlir::omp::ClauseNumTasksType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseNumTasksTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseNumTasksTypeAttrStorage>()) ClauseNumTasksTypeAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseNumTasksType value;
};
} // namespace detail
ClauseNumTasksTypeAttr ClauseNumTasksTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseNumTasksType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseNumTasksTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseNumTasksType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseNumTasksType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseNumTasksType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseNumTasksType" << " to be one of: " << "strict")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NumTasksTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseNumTasksType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseNumTasksTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseNumTasksType((*_result_value)));
}

void ClauseNumTasksTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseNumTasksType(getValue());
}

::mlir::omp::ClauseNumTasksType ClauseNumTasksTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseNumTasksTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseOrderKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseOrderKind>;
  ClauseOrderKindAttrStorage(::mlir::omp::ClauseOrderKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseOrderKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseOrderKindAttrStorage>()) ClauseOrderKindAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseOrderKind value;
};
} // namespace detail
ClauseOrderKindAttr ClauseOrderKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseOrderKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseOrderKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseOrderKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseOrderKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseOrderKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseOrderKind" << " to be one of: " << "concurrent")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OrderKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseOrderKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseOrderKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseOrderKind((*_result_value)));
}

void ClauseOrderKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseOrderKind(getValue());
}

::mlir::omp::ClauseOrderKind ClauseOrderKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseOrderKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseProcBindKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseProcBindKind>;
  ClauseProcBindKindAttrStorage(::mlir::omp::ClauseProcBindKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseProcBindKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseProcBindKindAttrStorage>()) ClauseProcBindKindAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseProcBindKind value;
};
} // namespace detail
ClauseProcBindKindAttr ClauseProcBindKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseProcBindKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseProcBindKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseProcBindKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseProcBindKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseProcBindKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseProcBindKind" << " to be one of: " << "primary" << ", " << "master" << ", " << "close" << ", " << "spread")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ProcBindKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseProcBindKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseProcBindKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseProcBindKind((*_result_value)));
}

void ClauseProcBindKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseProcBindKind(getValue());
}

::mlir::omp::ClauseProcBindKind ClauseProcBindKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseProcBindKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseScheduleKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseScheduleKind>;
  ClauseScheduleKindAttrStorage(::mlir::omp::ClauseScheduleKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseScheduleKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseScheduleKindAttrStorage>()) ClauseScheduleKindAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseScheduleKind value;
};
} // namespace detail
ClauseScheduleKindAttr ClauseScheduleKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseScheduleKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseScheduleKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseScheduleKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseScheduleKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseScheduleKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseScheduleKind" << " to be one of: " << "static" << ", " << "dynamic" << ", " << "guided" << ", " << "auto" << ", " << "runtime")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScheduleKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseScheduleKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseScheduleKindAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseScheduleKind((*_result_value)));
}

void ClauseScheduleKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseScheduleKind(getValue());
}

::mlir::omp::ClauseScheduleKind ClauseScheduleKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseScheduleKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct DeclareTargetCaptureClauseAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::DeclareTargetCaptureClause>;
  DeclareTargetCaptureClauseAttrStorage(::mlir::omp::DeclareTargetCaptureClause value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DeclareTargetCaptureClauseAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DeclareTargetCaptureClauseAttrStorage>()) DeclareTargetCaptureClauseAttrStorage(std::move(value));
  }

  ::mlir::omp::DeclareTargetCaptureClause value;
};
} // namespace detail
DeclareTargetCaptureClauseAttr DeclareTargetCaptureClauseAttr::get(::mlir::MLIRContext *context, ::mlir::omp::DeclareTargetCaptureClause value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DeclareTargetCaptureClauseAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::DeclareTargetCaptureClause> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::DeclareTargetCaptureClause> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeDeclareTargetCaptureClause(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::DeclareTargetCaptureClause" << " to be one of: " << "to" << ", " << "link" << ", " << "enter")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareTargetCaptureClauseAttr parameter 'value' which is to be a `::mlir::omp::DeclareTargetCaptureClause`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return DeclareTargetCaptureClauseAttr::get(odsParser.getContext(),
      ::mlir::omp::DeclareTargetCaptureClause((*_result_value)));
}

void DeclareTargetCaptureClauseAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyDeclareTargetCaptureClause(getValue());
  odsPrinter << ")";
}

::mlir::omp::DeclareTargetCaptureClause DeclareTargetCaptureClauseAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::DeclareTargetCaptureClauseAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseDependAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseDepend>;
  ClauseDependAttrStorage(::mlir::omp::ClauseDepend value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseDependAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseDependAttrStorage>()) ClauseDependAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseDepend value;
};
} // namespace detail
ClauseDependAttr ClauseDependAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseDepend value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseDependAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseDepend> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseDepend> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseDepend(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseDepend" << " to be one of: " << "dependsource" << ", " << "dependsink")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ClauseDependAttr parameter 'value' which is to be a `::mlir::omp::ClauseDepend`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return ClauseDependAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseDepend((*_result_value)));
}

void ClauseDependAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyClauseDepend(getValue());
  odsPrinter << ")";
}

::mlir::omp::ClauseDepend ClauseDependAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseDependAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseRequiresAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseRequires>;
  ClauseRequiresAttrStorage(::mlir::omp::ClauseRequires value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseRequiresAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseRequiresAttrStorage>()) ClauseRequiresAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseRequires value;
};
} // namespace detail
ClauseRequiresAttr ClauseRequiresAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseRequires value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseRequiresAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseRequires> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseRequires> {
      ::mlir::omp::ClauseRequires flags = {};
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      do {
        if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
          return ::mlir::failure();
        auto maybeEnum = ::mlir::omp::symbolizeClauseRequires(enumKeyword);
        if (!maybeEnum) {
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseRequires" << " to be one of: " << "none" << ", " << "reverse_offload" << ", " << "unified_address" << ", " << "unified_shared_memory" << ", " << "dynamic_allocators")};
        }
        flags = flags | *maybeEnum;
      } while(::mlir::succeeded(odsParser.parseOptionalVerticalBar()));
      return flags;
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ClauseRequiresAttr parameter 'value' which is to be a `::mlir::omp::ClauseRequires`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseRequiresAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseRequires((*_result_value)));
}

void ClauseRequiresAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseRequires(getValue());
}

::mlir::omp::ClauseRequires ClauseRequiresAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseRequiresAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseTaskDependAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseTaskDepend>;
  ClauseTaskDependAttrStorage(::mlir::omp::ClauseTaskDepend value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseTaskDependAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseTaskDependAttrStorage>()) ClauseTaskDependAttrStorage(std::move(value));
  }

  ::mlir::omp::ClauseTaskDepend value;
};
} // namespace detail
ClauseTaskDependAttr ClauseTaskDependAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseTaskDepend value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseTaskDependAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ClauseTaskDepend> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseTaskDepend> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseTaskDepend(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseTaskDepend" << " to be one of: " << "taskdependin" << ", " << "taskdependout" << ", " << "taskdependinout")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ClauseTaskDependAttr parameter 'value' which is to be a `::mlir::omp::ClauseTaskDepend`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return ClauseTaskDependAttr::get(odsParser.getContext(),
      ::mlir::omp::ClauseTaskDepend((*_result_value)));
}

void ClauseTaskDependAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyClauseTaskDepend(getValue());
  odsPrinter << ")";
}

::mlir::omp::ClauseTaskDepend ClauseTaskDependAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseTaskDependAttr)
namespace mlir {
namespace omp {
namespace detail {
struct DataSharingClauseTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::DataSharingClauseType>;
  DataSharingClauseTypeAttrStorage(::mlir::omp::DataSharingClauseType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DataSharingClauseTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DataSharingClauseTypeAttrStorage>()) DataSharingClauseTypeAttrStorage(std::move(value));
  }

  ::mlir::omp::DataSharingClauseType value;
};
} // namespace detail
DataSharingClauseTypeAttr DataSharingClauseTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::DataSharingClauseType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DataSharingClauseTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::DataSharingClauseType> _result_value;
  // Parse literal '{'
  if (odsParser.parseLBrace()) return {};
  // Parse literal 'type'
  if (odsParser.parseKeyword("type")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::DataSharingClauseType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeDataSharingClauseType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::DataSharingClauseType" << " to be one of: " << "private" << ", " << "firstprivate")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DataSharingClauseTypeAttr parameter 'value' which is to be a `::mlir::omp::DataSharingClauseType`");
    return {};
  }
  // Parse literal '}'
  if (odsParser.parseRBrace()) return {};
  assert(::mlir::succeeded(_result_value));
  return DataSharingClauseTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::DataSharingClauseType((*_result_value)));
}

void DataSharingClauseTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "{";
  odsPrinter << "type";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter << stringifyDataSharingClauseType(getValue());
  odsPrinter << "}";
}

::mlir::omp::DataSharingClauseType DataSharingClauseTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::DataSharingClauseTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct DeclareTargetDeviceTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::DeclareTargetDeviceType>;
  DeclareTargetDeviceTypeAttrStorage(::mlir::omp::DeclareTargetDeviceType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DeclareTargetDeviceTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DeclareTargetDeviceTypeAttrStorage>()) DeclareTargetDeviceTypeAttrStorage(std::move(value));
  }

  ::mlir::omp::DeclareTargetDeviceType value;
};
} // namespace detail
DeclareTargetDeviceTypeAttr DeclareTargetDeviceTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::DeclareTargetDeviceType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DeclareTargetDeviceTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::DeclareTargetDeviceType> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::DeclareTargetDeviceType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeDeclareTargetDeviceType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::DeclareTargetDeviceType" << " to be one of: " << "any" << ", " << "host" << ", " << "nohost")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareTargetDeviceTypeAttr parameter 'value' which is to be a `::mlir::omp::DeclareTargetDeviceType`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return DeclareTargetDeviceTypeAttr::get(odsParser.getContext(),
      ::mlir::omp::DeclareTargetDeviceType((*_result_value)));
}

void DeclareTargetDeviceTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyDeclareTargetDeviceType(getValue());
  odsPrinter << ")";
}

::mlir::omp::DeclareTargetDeviceType DeclareTargetDeviceTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::DeclareTargetDeviceTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct OrderModifierAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::OrderModifier>;
  OrderModifierAttrStorage(::mlir::omp::OrderModifier value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static OrderModifierAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<OrderModifierAttrStorage>()) OrderModifierAttrStorage(std::move(value));
  }

  ::mlir::omp::OrderModifier value;
};
} // namespace detail
OrderModifierAttr OrderModifierAttr::get(::mlir::MLIRContext *context, ::mlir::omp::OrderModifier value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute OrderModifierAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::OrderModifier> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::OrderModifier> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeOrderModifier(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::OrderModifier" << " to be one of: " << "reproducible" << ", " << "unconstrained")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OrderModifierAttr parameter 'value' which is to be a `::mlir::omp::OrderModifier`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return OrderModifierAttr::get(odsParser.getContext(),
      ::mlir::omp::OrderModifier((*_result_value)));
}

void OrderModifierAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyOrderModifier(getValue());
}

::mlir::omp::OrderModifier OrderModifierAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderModifierAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ScheduleModifierAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ScheduleModifier>;
  ScheduleModifierAttrStorage(::mlir::omp::ScheduleModifier value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ScheduleModifierAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ScheduleModifierAttrStorage>()) ScheduleModifierAttrStorage(std::move(value));
  }

  ::mlir::omp::ScheduleModifier value;
};
} // namespace detail
ScheduleModifierAttr ScheduleModifierAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ScheduleModifier value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ScheduleModifierAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::ScheduleModifier> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ScheduleModifier> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeScheduleModifier(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ScheduleModifier" << " to be one of: " << "none" << ", " << "monotonic" << ", " << "nonmonotonic" << ", " << "simd")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScheduleModifierAttr parameter 'value' which is to be a `::mlir::omp::ScheduleModifier`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ScheduleModifierAttr::get(odsParser.getContext(),
      ::mlir::omp::ScheduleModifier((*_result_value)));
}

void ScheduleModifierAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyScheduleModifier(getValue());
}

::mlir::omp::ScheduleModifier ScheduleModifierAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ScheduleModifierAttr)
namespace mlir {
namespace omp {
namespace detail {
struct VariableCaptureKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::VariableCaptureKind>;
  VariableCaptureKindAttrStorage(::mlir::omp::VariableCaptureKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static VariableCaptureKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<VariableCaptureKindAttrStorage>()) VariableCaptureKindAttrStorage(std::move(value));
  }

  ::mlir::omp::VariableCaptureKind value;
};
} // namespace detail
VariableCaptureKindAttr VariableCaptureKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::VariableCaptureKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute VariableCaptureKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::omp::VariableCaptureKind> _result_value;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::VariableCaptureKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeVariableCaptureKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::VariableCaptureKind" << " to be one of: " << "This" << ", " << "ByRef" << ", " << "ByCopy" << ", " << "VLAType")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VariableCaptureKindAttr parameter 'value' which is to be a `::mlir::omp::VariableCaptureKind`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return VariableCaptureKindAttr::get(odsParser.getContext(),
      ::mlir::omp::VariableCaptureKind((*_result_value)));
}

void VariableCaptureKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "(";
  odsPrinter << stringifyVariableCaptureKind(getValue());
  odsPrinter << ")";
}

::mlir::omp::VariableCaptureKind VariableCaptureKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::VariableCaptureKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct DeclareTargetAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<DeclareTargetDeviceTypeAttr, DeclareTargetCaptureClauseAttr>;
  DeclareTargetAttrStorage(DeclareTargetDeviceTypeAttr device_type, DeclareTargetCaptureClauseAttr capture_clause) : device_type(std::move(device_type)), capture_clause(std::move(capture_clause)) {}

  KeyTy getAsKey() const {
    return KeyTy(device_type, capture_clause);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (device_type == std::get<0>(tblgenKey)) && (capture_clause == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static DeclareTargetAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto device_type = std::move(std::get<0>(tblgenKey));
    auto capture_clause = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<DeclareTargetAttrStorage>()) DeclareTargetAttrStorage(std::move(device_type), std::move(capture_clause));
  }

  DeclareTargetDeviceTypeAttr device_type;
  DeclareTargetCaptureClauseAttr capture_clause;
};
} // namespace detail
DeclareTargetAttr DeclareTargetAttr::get(::mlir::MLIRContext *context, DeclareTargetDeviceTypeAttr device_type, DeclareTargetCaptureClauseAttr capture_clause) {
  return Base::get(context, std::move(device_type), std::move(capture_clause));
}

::mlir::Attribute DeclareTargetAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<DeclareTargetDeviceTypeAttr> _result_device_type;
  ::mlir::FailureOr<DeclareTargetCaptureClauseAttr> _result_capture_clause;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_device_type = false;
  bool _seen_capture_clause = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_device_type && _paramKey == "device_type") {
        _seen_device_type = true;

        // Parse variable 'device_type'
        _result_device_type = ::mlir::FieldParser<DeclareTargetDeviceTypeAttr>::parse(odsParser);
        if (::mlir::failed(_result_device_type)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareTargetAttr parameter 'device_type' which is to be a `DeclareTargetDeviceTypeAttr`");
          return {};
        }
      } else if (!_seen_capture_clause && _paramKey == "capture_clause") {
        _seen_capture_clause = true;

        // Parse variable 'capture_clause'
        _result_capture_clause = ::mlir::FieldParser<DeclareTargetCaptureClauseAttr>::parse(odsParser);
        if (::mlir::failed(_result_capture_clause)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareTargetAttr parameter 'capture_clause' which is to be a `DeclareTargetCaptureClauseAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return DeclareTargetAttr::get(odsParser.getContext(),
      DeclareTargetDeviceTypeAttr((_result_device_type.value_or(DeclareTargetDeviceTypeAttr()))),
      DeclareTargetCaptureClauseAttr((_result_capture_clause.value_or(DeclareTargetCaptureClauseAttr()))));
}

void DeclareTargetAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getDeviceType() == DeclareTargetDeviceTypeAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "device_type = ";
      if (!(getDeviceType() == DeclareTargetDeviceTypeAttr())) {
        odsPrinter.printStrippedAttrOrType(getDeviceType());
      }
    }
    if (!(getCaptureClause() == DeclareTargetCaptureClauseAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "capture_clause = ";
      if (!(getCaptureClause() == DeclareTargetCaptureClauseAttr())) {
        odsPrinter.printStrippedAttrOrType(getCaptureClause());
      }
    }
  }
  odsPrinter << ">";
}

DeclareTargetDeviceTypeAttr DeclareTargetAttr::getDeviceType() const {
  return getImpl()->device_type;
}

DeclareTargetCaptureClauseAttr DeclareTargetAttr::getCaptureClause() const {
  return getImpl()->capture_clause;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::DeclareTargetAttr)
namespace mlir {
namespace omp {
namespace detail {
struct FlagsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<uint32_t, bool, bool, bool, bool, bool, uint32_t>;
  FlagsAttrStorage(uint32_t debug_kind, bool assume_teams_oversubscription, bool assume_threads_oversubscription, bool assume_no_thread_state, bool assume_no_nested_parallelism, bool no_gpu_lib, uint32_t openmp_device_version) : debug_kind(std::move(debug_kind)), assume_teams_oversubscription(std::move(assume_teams_oversubscription)), assume_threads_oversubscription(std::move(assume_threads_oversubscription)), assume_no_thread_state(std::move(assume_no_thread_state)), assume_no_nested_parallelism(std::move(assume_no_nested_parallelism)), no_gpu_lib(std::move(no_gpu_lib)), openmp_device_version(std::move(openmp_device_version)) {}

  KeyTy getAsKey() const {
    return KeyTy(debug_kind, assume_teams_oversubscription, assume_threads_oversubscription, assume_no_thread_state, assume_no_nested_parallelism, no_gpu_lib, openmp_device_version);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (debug_kind == std::get<0>(tblgenKey)) && (assume_teams_oversubscription == std::get<1>(tblgenKey)) && (assume_threads_oversubscription == std::get<2>(tblgenKey)) && (assume_no_thread_state == std::get<3>(tblgenKey)) && (assume_no_nested_parallelism == std::get<4>(tblgenKey)) && (no_gpu_lib == std::get<5>(tblgenKey)) && (openmp_device_version == std::get<6>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
  }

  static FlagsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto debug_kind = std::move(std::get<0>(tblgenKey));
    auto assume_teams_oversubscription = std::move(std::get<1>(tblgenKey));
    auto assume_threads_oversubscription = std::move(std::get<2>(tblgenKey));
    auto assume_no_thread_state = std::move(std::get<3>(tblgenKey));
    auto assume_no_nested_parallelism = std::move(std::get<4>(tblgenKey));
    auto no_gpu_lib = std::move(std::get<5>(tblgenKey));
    auto openmp_device_version = std::move(std::get<6>(tblgenKey));
    return new (allocator.allocate<FlagsAttrStorage>()) FlagsAttrStorage(std::move(debug_kind), std::move(assume_teams_oversubscription), std::move(assume_threads_oversubscription), std::move(assume_no_thread_state), std::move(assume_no_nested_parallelism), std::move(no_gpu_lib), std::move(openmp_device_version));
  }

  uint32_t debug_kind;
  bool assume_teams_oversubscription;
  bool assume_threads_oversubscription;
  bool assume_no_thread_state;
  bool assume_no_nested_parallelism;
  bool no_gpu_lib;
  uint32_t openmp_device_version;
};
} // namespace detail
FlagsAttr FlagsAttr::get(::mlir::MLIRContext *context, uint32_t debug_kind, bool assume_teams_oversubscription, bool assume_threads_oversubscription, bool assume_no_thread_state, bool assume_no_nested_parallelism, bool no_gpu_lib, uint32_t openmp_device_version) {
  return Base::get(context, std::move(debug_kind), std::move(assume_teams_oversubscription), std::move(assume_threads_oversubscription), std::move(assume_no_thread_state), std::move(assume_no_nested_parallelism), std::move(no_gpu_lib), std::move(openmp_device_version));
}

::mlir::Attribute FlagsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<uint32_t> _result_debug_kind;
  ::mlir::FailureOr<bool> _result_assume_teams_oversubscription;
  ::mlir::FailureOr<bool> _result_assume_threads_oversubscription;
  ::mlir::FailureOr<bool> _result_assume_no_thread_state;
  ::mlir::FailureOr<bool> _result_assume_no_nested_parallelism;
  ::mlir::FailureOr<bool> _result_no_gpu_lib;
  ::mlir::FailureOr<uint32_t> _result_openmp_device_version;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_debug_kind = false;
  bool _seen_assume_teams_oversubscription = false;
  bool _seen_assume_threads_oversubscription = false;
  bool _seen_assume_no_thread_state = false;
  bool _seen_assume_no_nested_parallelism = false;
  bool _seen_no_gpu_lib = false;
  bool _seen_openmp_device_version = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_debug_kind && _paramKey == "debug_kind") {
        _seen_debug_kind = true;

        // Parse variable 'debug_kind'
        _result_debug_kind = ::mlir::FieldParser<uint32_t>::parse(odsParser);
        if (::mlir::failed(_result_debug_kind)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'debug_kind' which is to be a `uint32_t`");
          return {};
        }
      } else if (!_seen_assume_teams_oversubscription && _paramKey == "assume_teams_oversubscription") {
        _seen_assume_teams_oversubscription = true;

        // Parse variable 'assume_teams_oversubscription'
        _result_assume_teams_oversubscription = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_teams_oversubscription)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_teams_oversubscription' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_assume_threads_oversubscription && _paramKey == "assume_threads_oversubscription") {
        _seen_assume_threads_oversubscription = true;

        // Parse variable 'assume_threads_oversubscription'
        _result_assume_threads_oversubscription = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_threads_oversubscription)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_threads_oversubscription' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_assume_no_thread_state && _paramKey == "assume_no_thread_state") {
        _seen_assume_no_thread_state = true;

        // Parse variable 'assume_no_thread_state'
        _result_assume_no_thread_state = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_no_thread_state)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_no_thread_state' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_assume_no_nested_parallelism && _paramKey == "assume_no_nested_parallelism") {
        _seen_assume_no_nested_parallelism = true;

        // Parse variable 'assume_no_nested_parallelism'
        _result_assume_no_nested_parallelism = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_assume_no_nested_parallelism)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'assume_no_nested_parallelism' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_no_gpu_lib && _paramKey == "no_gpu_lib") {
        _seen_no_gpu_lib = true;

        // Parse variable 'no_gpu_lib'
        _result_no_gpu_lib = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_no_gpu_lib)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'no_gpu_lib' which is to be a `bool`");
          return {};
        }
      } else if (!_seen_openmp_device_version && _paramKey == "openmp_device_version") {
        _seen_openmp_device_version = true;

        // Parse variable 'openmp_device_version'
        _result_openmp_device_version = ::mlir::FieldParser<uint32_t>::parse(odsParser);
        if (::mlir::failed(_result_openmp_device_version)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FlagsAttr parameter 'openmp_device_version' which is to be a `uint32_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return FlagsAttr::get(odsParser.getContext(),
      uint32_t((_result_debug_kind.value_or(0))),
      bool((_result_assume_teams_oversubscription.value_or(false))),
      bool((_result_assume_threads_oversubscription.value_or(false))),
      bool((_result_assume_no_thread_state.value_or(false))),
      bool((_result_assume_no_nested_parallelism.value_or(false))),
      bool((_result_no_gpu_lib.value_or(false))),
      uint32_t((_result_openmp_device_version.value_or(50))));
}

void FlagsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getDebugKind() == 0)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "debug_kind = ";
      if (!(getDebugKind() == 0)) {
        odsPrinter.printStrippedAttrOrType(getDebugKind());
      }
    }
    if (!(getAssumeTeamsOversubscription() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_teams_oversubscription = ";
      if (!(getAssumeTeamsOversubscription() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeTeamsOversubscription());
      }
    }
    if (!(getAssumeThreadsOversubscription() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_threads_oversubscription = ";
      if (!(getAssumeThreadsOversubscription() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeThreadsOversubscription());
      }
    }
    if (!(getAssumeNoThreadState() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_no_thread_state = ";
      if (!(getAssumeNoThreadState() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeNoThreadState());
      }
    }
    if (!(getAssumeNoNestedParallelism() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "assume_no_nested_parallelism = ";
      if (!(getAssumeNoNestedParallelism() == false)) {
        odsPrinter.printStrippedAttrOrType(getAssumeNoNestedParallelism());
      }
    }
    if (!(getNoGpuLib() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "no_gpu_lib = ";
      if (!(getNoGpuLib() == false)) {
        odsPrinter.printStrippedAttrOrType(getNoGpuLib());
      }
    }
    if (!(getOpenmpDeviceVersion() == 50)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "openmp_device_version = ";
      if (!(getOpenmpDeviceVersion() == 50)) {
        odsPrinter.printStrippedAttrOrType(getOpenmpDeviceVersion());
      }
    }
  }
  odsPrinter << ">";
}

uint32_t FlagsAttr::getDebugKind() const {
  return getImpl()->debug_kind;
}

bool FlagsAttr::getAssumeTeamsOversubscription() const {
  return getImpl()->assume_teams_oversubscription;
}

bool FlagsAttr::getAssumeThreadsOversubscription() const {
  return getImpl()->assume_threads_oversubscription;
}

bool FlagsAttr::getAssumeNoThreadState() const {
  return getImpl()->assume_no_thread_state;
}

bool FlagsAttr::getAssumeNoNestedParallelism() const {
  return getImpl()->assume_no_nested_parallelism;
}

bool FlagsAttr::getNoGpuLib() const {
  return getImpl()->no_gpu_lib;
}

uint32_t FlagsAttr::getOpenmpDeviceVersion() const {
  return getImpl()->openmp_device_version;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::FlagsAttr)
namespace mlir {
namespace omp {
namespace detail {
struct VersionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<uint32_t>;
  VersionAttrStorage(uint32_t version) : version(std::move(version)) {}

  KeyTy getAsKey() const {
    return KeyTy(version);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (version == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static VersionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto version = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<VersionAttrStorage>()) VersionAttrStorage(std::move(version));
  }

  uint32_t version;
};
} // namespace detail
VersionAttr VersionAttr::get(::mlir::MLIRContext *context, uint32_t version) {
  return Base::get(context, std::move(version));
}

::mlir::Attribute VersionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<uint32_t> _result_version;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_version = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_version && _paramKey == "version") {
        _seen_version = true;

        // Parse variable 'version'
        _result_version = ::mlir::FieldParser<uint32_t>::parse(odsParser);
        if (::mlir::failed(_result_version)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VersionAttr parameter 'version' which is to be a `uint32_t`");
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
  assert(::mlir::succeeded(_result_version));
  return VersionAttr::get(odsParser.getContext(),
      uint32_t((*_result_version)));
}

void VersionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "version = ";
    odsPrinter.printStrippedAttrOrType(getVersion());
  }
  odsPrinter << ">";
}

uint32_t VersionAttr::getVersion() const {
  return getImpl()->version;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::VersionAttr)
namespace mlir {
namespace omp {

/// Parse an attribute registered to this dialect.
::mlir::Attribute OpenMPDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void OpenMPDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace omp
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

