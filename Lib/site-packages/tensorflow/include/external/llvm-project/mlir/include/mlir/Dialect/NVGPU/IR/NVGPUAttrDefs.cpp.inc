/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::nvgpu::TensorMapSwizzleKindAttr,
::mlir::nvgpu::TensorMapL2PromoKindAttr,
::mlir::nvgpu::TensorMapOOBKindAttr,
::mlir::nvgpu::TensorMapInterleaveKindAttr,
::mlir::nvgpu::RcpRoundingModeAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::nvgpu::TensorMapSwizzleKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::TensorMapSwizzleKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::TensorMapL2PromoKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::TensorMapL2PromoKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::TensorMapOOBKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::TensorMapOOBKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::TensorMapInterleaveKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::TensorMapInterleaveKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::nvgpu::RcpRoundingModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::nvgpu::RcpRoundingModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::nvgpu::TensorMapSwizzleKindAttr>([&](auto t) {
      printer << ::mlir::nvgpu::TensorMapSwizzleKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::TensorMapL2PromoKindAttr>([&](auto t) {
      printer << ::mlir::nvgpu::TensorMapL2PromoKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::TensorMapOOBKindAttr>([&](auto t) {
      printer << ::mlir::nvgpu::TensorMapOOBKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::TensorMapInterleaveKindAttr>([&](auto t) {
      printer << ::mlir::nvgpu::TensorMapInterleaveKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::nvgpu::RcpRoundingModeAttr>([&](auto t) {
      printer << ::mlir::nvgpu::RcpRoundingModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace nvgpu {
namespace detail {
struct TensorMapSwizzleKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::nvgpu::TensorMapSwizzleKind>;
  TensorMapSwizzleKindAttrStorage(::mlir::nvgpu::TensorMapSwizzleKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TensorMapSwizzleKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TensorMapSwizzleKindAttrStorage>()) TensorMapSwizzleKindAttrStorage(std::move(value));
  }

  ::mlir::nvgpu::TensorMapSwizzleKind value;
};
} // namespace detail
TensorMapSwizzleKindAttr TensorMapSwizzleKindAttr::get(::mlir::MLIRContext *context, ::mlir::nvgpu::TensorMapSwizzleKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TensorMapSwizzleKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapSwizzleKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapSwizzleKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapSwizzleKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapSwizzleKind" << " to be one of: " << "none" << ", " << "swizzle_32b" << ", " << "swizzle_64b" << ", " << "swizzle_128b")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TensorMapSwizzleAttr parameter 'value' which is to be a `::mlir::nvgpu::TensorMapSwizzleKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TensorMapSwizzleKindAttr::get(odsParser.getContext(),
      ::mlir::nvgpu::TensorMapSwizzleKind((*_result_value)));
}

void TensorMapSwizzleKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTensorMapSwizzleKind(getValue());
}

::mlir::nvgpu::TensorMapSwizzleKind TensorMapSwizzleKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::TensorMapSwizzleKindAttr)
namespace mlir {
namespace nvgpu {
namespace detail {
struct TensorMapL2PromoKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::nvgpu::TensorMapL2PromoKind>;
  TensorMapL2PromoKindAttrStorage(::mlir::nvgpu::TensorMapL2PromoKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TensorMapL2PromoKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TensorMapL2PromoKindAttrStorage>()) TensorMapL2PromoKindAttrStorage(std::move(value));
  }

  ::mlir::nvgpu::TensorMapL2PromoKind value;
};
} // namespace detail
TensorMapL2PromoKindAttr TensorMapL2PromoKindAttr::get(::mlir::MLIRContext *context, ::mlir::nvgpu::TensorMapL2PromoKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TensorMapL2PromoKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapL2PromoKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapL2PromoKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapL2PromoKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapL2PromoKind" << " to be one of: " << "none" << ", " << "l2promo_64b" << ", " << "l2promo_128b" << ", " << "l2promo_256b")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TensorMapL2PromoAttr parameter 'value' which is to be a `::mlir::nvgpu::TensorMapL2PromoKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TensorMapL2PromoKindAttr::get(odsParser.getContext(),
      ::mlir::nvgpu::TensorMapL2PromoKind((*_result_value)));
}

void TensorMapL2PromoKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTensorMapL2PromoKind(getValue());
}

::mlir::nvgpu::TensorMapL2PromoKind TensorMapL2PromoKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::TensorMapL2PromoKindAttr)
namespace mlir {
namespace nvgpu {
namespace detail {
struct TensorMapOOBKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::nvgpu::TensorMapOOBKind>;
  TensorMapOOBKindAttrStorage(::mlir::nvgpu::TensorMapOOBKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TensorMapOOBKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TensorMapOOBKindAttrStorage>()) TensorMapOOBKindAttrStorage(std::move(value));
  }

  ::mlir::nvgpu::TensorMapOOBKind value;
};
} // namespace detail
TensorMapOOBKindAttr TensorMapOOBKindAttr::get(::mlir::MLIRContext *context, ::mlir::nvgpu::TensorMapOOBKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TensorMapOOBKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapOOBKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapOOBKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapOOBKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapOOBKind" << " to be one of: " << "zero" << ", " << "nan")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TensorMapOOBAttr parameter 'value' which is to be a `::mlir::nvgpu::TensorMapOOBKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TensorMapOOBKindAttr::get(odsParser.getContext(),
      ::mlir::nvgpu::TensorMapOOBKind((*_result_value)));
}

void TensorMapOOBKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTensorMapOOBKind(getValue());
}

::mlir::nvgpu::TensorMapOOBKind TensorMapOOBKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::TensorMapOOBKindAttr)
namespace mlir {
namespace nvgpu {
namespace detail {
struct TensorMapInterleaveKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::nvgpu::TensorMapInterleaveKind>;
  TensorMapInterleaveKindAttrStorage(::mlir::nvgpu::TensorMapInterleaveKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TensorMapInterleaveKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TensorMapInterleaveKindAttrStorage>()) TensorMapInterleaveKindAttrStorage(std::move(value));
  }

  ::mlir::nvgpu::TensorMapInterleaveKind value;
};
} // namespace detail
TensorMapInterleaveKindAttr TensorMapInterleaveKindAttr::get(::mlir::MLIRContext *context, ::mlir::nvgpu::TensorMapInterleaveKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TensorMapInterleaveKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::nvgpu::TensorMapInterleaveKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::TensorMapInterleaveKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::nvgpu::symbolizeTensorMapInterleaveKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::TensorMapInterleaveKind" << " to be one of: " << "none" << ", " << "interleave_16b" << ", " << "interleave_32b")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TensorMapInterleaveAttr parameter 'value' which is to be a `::mlir::nvgpu::TensorMapInterleaveKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TensorMapInterleaveKindAttr::get(odsParser.getContext(),
      ::mlir::nvgpu::TensorMapInterleaveKind((*_result_value)));
}

void TensorMapInterleaveKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTensorMapInterleaveKind(getValue());
}

::mlir::nvgpu::TensorMapInterleaveKind TensorMapInterleaveKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::TensorMapInterleaveKindAttr)
namespace mlir {
namespace nvgpu {
namespace detail {
struct RcpRoundingModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::nvgpu::RcpRoundingMode>;
  RcpRoundingModeAttrStorage(::mlir::nvgpu::RcpRoundingMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RcpRoundingModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RcpRoundingModeAttrStorage>()) RcpRoundingModeAttrStorage(std::move(value));
  }

  ::mlir::nvgpu::RcpRoundingMode value;
};
} // namespace detail
RcpRoundingModeAttr RcpRoundingModeAttr::get(::mlir::MLIRContext *context, ::mlir::nvgpu::RcpRoundingMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RcpRoundingModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::nvgpu::RcpRoundingMode> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::nvgpu::RcpRoundingMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::nvgpu::symbolizeRcpRoundingMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::nvgpu::RcpRoundingMode" << " to be one of: " << "approx" << ", " << "rn" << ", " << "rz" << ", " << "rm" << ", " << "rp")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse RcpRoundingModeAttr parameter 'value' which is to be a `::mlir::nvgpu::RcpRoundingMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RcpRoundingModeAttr::get(odsParser.getContext(),
      ::mlir::nvgpu::RcpRoundingMode((*_result_value)));
}

void RcpRoundingModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRcpRoundingMode(getValue());
}

::mlir::nvgpu::RcpRoundingMode RcpRoundingModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::RcpRoundingModeAttr)
namespace mlir {
namespace nvgpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute NVGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void NVGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace nvgpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

