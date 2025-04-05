/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::DataLayoutEntryAttr,
::mlir::DataLayoutSpecAttr,
::mlir::MapAttr,
::mlir::TargetSystemSpecAttr,
::mlir::TargetDeviceSpecAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::DataLayoutEntryAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::DataLayoutEntryAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::DataLayoutSpecAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::DataLayoutSpecAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::MapAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::MapAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::TargetSystemSpecAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::TargetSystemSpecAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::TargetDeviceSpecAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::TargetDeviceSpecAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::DataLayoutEntryAttr>([&](auto t) {
      printer << ::mlir::DataLayoutEntryAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::DataLayoutSpecAttr>([&](auto t) {
      printer << ::mlir::DataLayoutSpecAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::MapAttr>([&](auto t) {
      printer << ::mlir::MapAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::TargetSystemSpecAttr>([&](auto t) {
      printer << ::mlir::TargetSystemSpecAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::TargetDeviceSpecAttr>([&](auto t) {
      printer << ::mlir::TargetDeviceSpecAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
DataLayoutEntryAttr DataLayoutEntryAttr::get(::mlir::MLIRContext *context, DataLayoutEntryKey key, Attribute value) {
  return Base::get(context, std::move(key), std::move(value));
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DataLayoutEntryAttr)
namespace mlir {
namespace detail {
struct DataLayoutSpecAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<DataLayoutEntryInterface>>;
  DataLayoutSpecAttrStorage(::llvm::ArrayRef<DataLayoutEntryInterface> entries) : entries(std::move(entries)) {}

  KeyTy getAsKey() const {
    return KeyTy(entries);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (entries == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DataLayoutSpecAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto entries = std::move(std::get<0>(tblgenKey));
    entries = allocator.copyInto(entries);
    return new (allocator.allocate<DataLayoutSpecAttrStorage>()) DataLayoutSpecAttrStorage(std::move(entries));
  }

  ::llvm::ArrayRef<DataLayoutEntryInterface> entries;
};
} // namespace detail
DataLayoutSpecAttr DataLayoutSpecAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::get(context, std::move(entries));
}

DataLayoutSpecAttr DataLayoutSpecAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::getChecked(emitError, context, entries);
}

::llvm::LogicalResult DataLayoutSpecAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  if (::mlir::failed(verify(emitError, entries)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<DataLayoutEntryInterface> DataLayoutSpecAttr::getEntries() const {
  return getImpl()->entries;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DataLayoutSpecAttr)
namespace mlir {
namespace detail {
struct MapAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<DataLayoutEntryInterface>>;
  MapAttrStorage(::llvm::ArrayRef<DataLayoutEntryInterface> entries) : entries(std::move(entries)) {}

  KeyTy getAsKey() const {
    return KeyTy(entries);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (entries == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MapAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto entries = std::move(std::get<0>(tblgenKey));
    entries = allocator.copyInto(entries);
    return new (allocator.allocate<MapAttrStorage>()) MapAttrStorage(std::move(entries));
  }

  ::llvm::ArrayRef<DataLayoutEntryInterface> entries;
};
} // namespace detail
MapAttr MapAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::get(context, std::move(entries));
}

MapAttr MapAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::getChecked(emitError, context, entries);
}

::llvm::LogicalResult MapAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  if (::mlir::failed(verify(emitError, entries)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute MapAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<DataLayoutEntryInterface>> _result_entries;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'entries'
  _result_entries = ::mlir::FieldParser<::llvm::SmallVector<DataLayoutEntryInterface>>::parse(odsParser);
  if (::mlir::failed(_result_entries)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DLTI_MapAttr parameter 'entries' which is to be a `::llvm::ArrayRef<DataLayoutEntryInterface>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_entries));
  return odsParser.getChecked<MapAttr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<DataLayoutEntryInterface>((*_result_entries)));
}

void MapAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getEntries());
  odsPrinter << ">";
}

::llvm::ArrayRef<DataLayoutEntryInterface> MapAttr::getEntries() const {
  return getImpl()->entries;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::MapAttr)
namespace mlir {
namespace detail {
struct TargetSystemSpecAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair>>;
  TargetSystemSpecAttrStorage(::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries) : entries(std::move(entries)) {}

  KeyTy getAsKey() const {
    return KeyTy(entries);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (entries == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TargetSystemSpecAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto entries = std::move(std::get<0>(tblgenKey));
    entries = allocator.copyInto(entries);
    return new (allocator.allocate<TargetSystemSpecAttrStorage>()) TargetSystemSpecAttrStorage(std::move(entries));
  }

  ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries;
};
} // namespace detail
std::optional<TargetDeviceSpecInterface>
TargetSystemSpecAttr::getDeviceSpecForDeviceID(
    TargetSystemSpecInterface::DeviceID deviceID) {
  for (const auto& entry : getEntries()) {
    if (entry.first == deviceID)
      return entry.second;
  }
  return std::nullopt;
}
TargetSystemSpecAttr TargetSystemSpecAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries) {
  return Base::get(context, std::move(entries));
}

TargetSystemSpecAttr TargetSystemSpecAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries) {
  return Base::getChecked(emitError, context, entries);
}

::llvm::LogicalResult TargetSystemSpecAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> entries) {
  if (::mlir::failed(verify(emitError, entries)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute TargetSystemSpecAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<DeviceIDTargetDeviceSpecPair>> _result_entries;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'entries'
  _result_entries = ::mlir::FieldParser<::llvm::SmallVector<DeviceIDTargetDeviceSpecPair>>::parse(odsParser);
  if (::mlir::failed(_result_entries)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DLTI_TargetSystemSpecAttr parameter 'entries' which is to be a `::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_entries));
  return odsParser.getChecked<TargetSystemSpecAttr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair>((*_result_entries)));
}

void TargetSystemSpecAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getEntries());
  odsPrinter << ">";
}

::llvm::ArrayRef<DeviceIDTargetDeviceSpecPair> TargetSystemSpecAttr::getEntries() const {
  return getImpl()->entries;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TargetSystemSpecAttr)
namespace mlir {
namespace detail {
struct TargetDeviceSpecAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<DataLayoutEntryInterface>>;
  TargetDeviceSpecAttrStorage(::llvm::ArrayRef<DataLayoutEntryInterface> entries) : entries(std::move(entries)) {}

  KeyTy getAsKey() const {
    return KeyTy(entries);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (entries == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TargetDeviceSpecAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto entries = std::move(std::get<0>(tblgenKey));
    entries = allocator.copyInto(entries);
    return new (allocator.allocate<TargetDeviceSpecAttrStorage>()) TargetDeviceSpecAttrStorage(std::move(entries));
  }

  ::llvm::ArrayRef<DataLayoutEntryInterface> entries;
};
} // namespace detail
TargetDeviceSpecAttr TargetDeviceSpecAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::get(context, std::move(entries));
}

TargetDeviceSpecAttr TargetDeviceSpecAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  return Base::getChecked(emitError, context, entries);
}

::llvm::LogicalResult TargetDeviceSpecAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<DataLayoutEntryInterface> entries) {
  if (::mlir::failed(verify(emitError, entries)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute TargetDeviceSpecAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<DataLayoutEntryInterface>> _result_entries;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'entries'
  _result_entries = ::mlir::FieldParser<::llvm::SmallVector<DataLayoutEntryInterface>>::parse(odsParser);
  if (::mlir::failed(_result_entries)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DLTI_TargetDeviceSpecAttr parameter 'entries' which is to be a `::llvm::ArrayRef<DataLayoutEntryInterface>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_entries));
  return odsParser.getChecked<TargetDeviceSpecAttr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<DataLayoutEntryInterface>((*_result_entries)));
}

void TargetDeviceSpecAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getEntries());
  odsPrinter << ">";
}

::llvm::ArrayRef<DataLayoutEntryInterface> TargetDeviceSpecAttr::getEntries() const {
  return getImpl()->entries;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TargetDeviceSpecAttr)
namespace mlir {

/// Parse an attribute registered to this dialect.
::mlir::Attribute DLTIDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void DLTIDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

