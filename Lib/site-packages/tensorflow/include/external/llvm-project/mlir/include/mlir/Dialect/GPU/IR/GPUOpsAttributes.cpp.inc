/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::gpu::AddressSpaceAttr,
::mlir::gpu::KernelMetadataAttr,
::mlir::gpu::KernelTableAttr,
::mlir::gpu::ObjectAttr,
::mlir::gpu::SelectObjectAttr,
::mlir::gpu::GPUBlockMappingAttr,
::mlir::gpu::GPUWarpgroupMappingAttr,
::mlir::gpu::GPUWarpMappingAttr,
::mlir::gpu::GPUThreadMappingAttr,
::mlir::gpu::GPUMemorySpaceMappingAttr,
::mlir::gpu::ParallelLoopDimMappingAttr,
::mlir::gpu::DimensionAttr,
::mlir::gpu::AllReduceOperationAttr,
::mlir::gpu::ShuffleModeAttr,
::mlir::gpu::MMAElementwiseOpAttr,
::mlir::gpu::Prune2To4SpMatFlagAttr,
::mlir::gpu::TransposeModeAttr,
::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::gpu::AddressSpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::AddressSpaceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::KernelMetadataAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::KernelMetadataAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::KernelTableAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::KernelTableAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ObjectAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ObjectAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::SelectObjectAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::SelectObjectAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUBlockMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUBlockMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUWarpgroupMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUWarpgroupMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUWarpMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUWarpMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUThreadMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUThreadMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::GPUMemorySpaceMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::GPUMemorySpaceMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ParallelLoopDimMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ParallelLoopDimMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::DimensionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::DimensionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::AllReduceOperationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::AllReduceOperationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ShuffleModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ShuffleModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::MMAElementwiseOpAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::MMAElementwiseOpAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::Prune2To4SpMatFlagAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::Prune2To4SpMatFlagAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::TransposeModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::TransposeModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::gpu::AddressSpaceAttr>([&](auto t) {
      printer << ::mlir::gpu::AddressSpaceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::KernelMetadataAttr>([&](auto t) {
      printer << ::mlir::gpu::KernelMetadataAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::KernelTableAttr>([&](auto t) {
      printer << ::mlir::gpu::KernelTableAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ObjectAttr>([&](auto t) {
      printer << ::mlir::gpu::ObjectAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::SelectObjectAttr>([&](auto t) {
      printer << ::mlir::gpu::SelectObjectAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUBlockMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUBlockMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUWarpgroupMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUWarpgroupMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUWarpMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUWarpMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUThreadMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUThreadMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::GPUMemorySpaceMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::GPUMemorySpaceMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ParallelLoopDimMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::ParallelLoopDimMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::DimensionAttr>([&](auto t) {
      printer << ::mlir::gpu::DimensionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::AllReduceOperationAttr>([&](auto t) {
      printer << ::mlir::gpu::AllReduceOperationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ShuffleModeAttr>([&](auto t) {
      printer << ::mlir::gpu::ShuffleModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::MMAElementwiseOpAttr>([&](auto t) {
      printer << ::mlir::gpu::MMAElementwiseOpAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::Prune2To4SpMatFlagAttr>([&](auto t) {
      printer << ::mlir::gpu::Prune2To4SpMatFlagAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::TransposeModeAttr>([&](auto t) {
      printer << ::mlir::gpu::TransposeModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr>([&](auto t) {
      printer << ::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace gpu {
namespace detail {
struct AddressSpaceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AddressSpace>;
  AddressSpaceAttrStorage(::mlir::gpu::AddressSpace value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AddressSpaceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<AddressSpaceAttrStorage>()) AddressSpaceAttrStorage(std::move(value));
  }

  ::mlir::gpu::AddressSpace value;
};
} // namespace detail
AddressSpaceAttr AddressSpaceAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute AddressSpaceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::AddressSpace> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::AddressSpace> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAddressSpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AddressSpace" << " to be one of: " << "global" << ", " << "workgroup" << ", " << "private")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_AddressSpaceAttr parameter 'value' which is to be a `::mlir::gpu::AddressSpace`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return AddressSpaceAttr::get(odsParser.getContext(),
      ::mlir::gpu::AddressSpace((*_result_value)));
}

void AddressSpaceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyAddressSpace(getValue());
  odsPrinter << ">";
}

::mlir::gpu::AddressSpace AddressSpaceAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AddressSpaceAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct KernelMetadataAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, Type, ArrayAttr, DictionaryAttr>;
  KernelMetadataAttrStorage(StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata) : name(std::move(name)), function_type(std::move(function_type)), arg_attrs(std::move(arg_attrs)), metadata(std::move(metadata)) {}

  KeyTy getAsKey() const {
    return KeyTy(name, function_type, arg_attrs, metadata);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey)) && (function_type == std::get<1>(tblgenKey)) && (arg_attrs == std::get<2>(tblgenKey)) && (metadata == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static KernelMetadataAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto name = std::move(std::get<0>(tblgenKey));
    auto function_type = std::move(std::get<1>(tblgenKey));
    auto arg_attrs = std::move(std::get<2>(tblgenKey));
    auto metadata = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<KernelMetadataAttrStorage>()) KernelMetadataAttrStorage(std::move(name), std::move(function_type), std::move(arg_attrs), std::move(metadata));
  }

  StringAttr name;
  Type function_type;
  ArrayAttr arg_attrs;
  DictionaryAttr metadata;
};
} // namespace detail
KernelMetadataAttr KernelMetadataAttr::get(::mlir::MLIRContext *context, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata) {
  return Base::get(context, std::move(name), std::move(function_type), std::move(arg_attrs), std::move(metadata));
}

KernelMetadataAttr KernelMetadataAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata) {
  return Base::getChecked(emitError, context, name, function_type, arg_attrs, metadata);
}

KernelMetadataAttr KernelMetadataAttr::get(StringAttr name, Type functionType, ArrayAttr argAttrs, DictionaryAttr metadata) {
  assert(name && "invalid name");
  return Base::get(name.getContext(), name, functionType, argAttrs, metadata);
}

KernelMetadataAttr KernelMetadataAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr name, Type functionType, ArrayAttr argAttrs, DictionaryAttr metadata) {
  assert(name && "invalid name");
  return Base::getChecked(emitError, name.getContext(), name, functionType, argAttrs, metadata);
}

::llvm::LogicalResult KernelMetadataAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr name, Type function_type, ArrayAttr arg_attrs, DictionaryAttr metadata) {
  if (::mlir::failed(verify(emitError, name, function_type, arg_attrs, metadata)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute KernelMetadataAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_name;
  ::mlir::FailureOr<Type> _result_function_type;
  ::mlir::FailureOr<ArrayAttr> _result_arg_attrs;
  ::mlir::FailureOr<DictionaryAttr> _result_metadata;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_KernelMetadataAttr parameter 'name' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'function_type'
  _result_function_type = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_function_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_KernelMetadataAttr parameter 'function_type' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse parameter struct
    bool _seen_arg_attrs = false;
    bool _seen_metadata = false;
    {
      const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
        // Parse literal '='
        if (odsParser.parseEqual()) return {};
        if (!_seen_arg_attrs && _paramKey == "arg_attrs") {
          _seen_arg_attrs = true;

          // Parse variable 'arg_attrs'
          _result_arg_attrs = ::mlir::FieldParser<ArrayAttr>::parse(odsParser);
          if (::mlir::failed(_result_arg_attrs)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_KernelMetadataAttr parameter 'arg_attrs' which is to be a `ArrayAttr`");
            return {};
          }
        } else if (!_seen_metadata && _paramKey == "metadata") {
          _seen_metadata = true;

          // Parse variable 'metadata'
          _result_metadata = ::mlir::FieldParser<DictionaryAttr>::parse(odsParser);
          if (::mlir::failed(_result_metadata)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_KernelMetadataAttr parameter 'metadata' which is to be a `DictionaryAttr`");
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
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  assert(::mlir::succeeded(_result_function_type));
  return odsParser.getChecked<KernelMetadataAttr>(odsLoc, odsParser.getContext(),
      StringAttr((*_result_name)),
      Type((*_result_function_type)),
      ArrayAttr((_result_arg_attrs.value_or(ArrayAttr()))),
      DictionaryAttr((_result_metadata.value_or(DictionaryAttr()))));
}

void KernelMetadataAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getName());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getFunctionType());
  if (!(getArgAttrs() == ArrayAttr()) || !(getMetadata() == DictionaryAttr())) {
    odsPrinter << ",";
    if (!(getArgAttrs() == ArrayAttr()) || !(getMetadata() == DictionaryAttr())) {
      odsPrinter << ' ';
    }
    {
      bool _firstPrinted = true;
      if (!(getArgAttrs() == ArrayAttr())) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "arg_attrs = ";
        if (!(getArgAttrs() == ArrayAttr())) {
          odsPrinter.printStrippedAttrOrType(getArgAttrs());
        }
      }
      if (!(getMetadata() == DictionaryAttr())) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "metadata = ";
        if (!(getMetadata() == DictionaryAttr())) {
          odsPrinter.printStrippedAttrOrType(getMetadata());
        }
      }
    }
  } else {
  }
  odsPrinter << ">";
}

StringAttr KernelMetadataAttr::getName() const {
  return getImpl()->name;
}

Type KernelMetadataAttr::getFunctionType() const {
  return getImpl()->function_type;
}

ArrayAttr KernelMetadataAttr::getArgAttrs() const {
  return getImpl()->arg_attrs;
}

DictionaryAttr KernelMetadataAttr::getMetadata() const {
  return getImpl()->metadata;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::KernelMetadataAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct KernelTableAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<KernelMetadataAttr>>;
  KernelTableAttrStorage(::llvm::ArrayRef<KernelMetadataAttr> kernel_table) : kernel_table(std::move(kernel_table)) {}

  KeyTy getAsKey() const {
    return KeyTy(kernel_table);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (::llvm::ArrayRef<KernelMetadataAttr>(kernel_table) == ::llvm::ArrayRef<KernelMetadataAttr>(std::get<0>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static KernelTableAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto kernel_table = std::move(std::get<0>(tblgenKey));
    kernel_table = allocator.copyInto(kernel_table);
    return new (allocator.allocate<KernelTableAttrStorage>()) KernelTableAttrStorage(std::move(kernel_table));
  }

  ::llvm::ArrayRef<KernelMetadataAttr> kernel_table;
};
} // namespace detail
::llvm::LogicalResult KernelTableAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<KernelMetadataAttr> kernel_table) {
  if (::mlir::failed(verify(emitError, kernel_table)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute KernelTableAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<KernelMetadataAttr>> _result_kernel_table;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '['
  if (odsParser.parseOptionalLSquare()) {
  } else {

    // Parse variable 'kernel_table'
    _result_kernel_table = ::mlir::FieldParser<::llvm::SmallVector<KernelMetadataAttr>>::parse(odsParser);
    if (::mlir::failed(_result_kernel_table)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_KernelTableAttr parameter 'kernel_table' which is to be a `::llvm::ArrayRef<KernelMetadataAttr>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return odsParser.getChecked<KernelTableAttr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<KernelMetadataAttr>((_result_kernel_table.value_or(::llvm::SmallVector<KernelMetadataAttr>()))));
}

void KernelTableAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  if (!(::llvm::ArrayRef<KernelMetadataAttr>(getKernelTable()) == ::llvm::ArrayRef<KernelMetadataAttr>(::llvm::SmallVector<KernelMetadataAttr>()))) {
    odsPrinter << "[";
    if (!(::llvm::ArrayRef<KernelMetadataAttr>(getKernelTable()) == ::llvm::ArrayRef<KernelMetadataAttr>(::llvm::SmallVector<KernelMetadataAttr>()))) {
      odsPrinter << getKernelTable();
    }
    odsPrinter << "]";
  } else {
  }
  odsPrinter << ">";
}

::llvm::ArrayRef<KernelMetadataAttr> KernelTableAttr::getKernelTable() const {
  return getImpl()->kernel_table;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::KernelTableAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ObjectAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Attribute, CompilationTarget, StringAttr, DictionaryAttr, KernelTableAttr>;
  ObjectAttrStorage(Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels) : target(std::move(target)), format(std::move(format)), object(std::move(object)), properties(std::move(properties)), kernels(std::move(kernels)) {}

  KeyTy getAsKey() const {
    return KeyTy(target, format, object, properties, kernels);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (target == std::get<0>(tblgenKey)) && (format == std::get<1>(tblgenKey)) && (object == std::get<2>(tblgenKey)) && (properties == std::get<3>(tblgenKey)) && (kernels == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static ObjectAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto target = std::move(std::get<0>(tblgenKey));
    auto format = std::move(std::get<1>(tblgenKey));
    auto object = std::move(std::get<2>(tblgenKey));
    auto properties = std::move(std::get<3>(tblgenKey));
    auto kernels = std::move(std::get<4>(tblgenKey));
    return new (allocator.allocate<ObjectAttrStorage>()) ObjectAttrStorage(std::move(target), std::move(format), std::move(object), std::move(properties), std::move(kernels));
  }

  Attribute target;
  CompilationTarget format;
  StringAttr object;
  DictionaryAttr properties;
  KernelTableAttr kernels;
};
} // namespace detail
ObjectAttr ObjectAttr::get(::mlir::MLIRContext *context, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels) {
  return Base::get(context, std::move(target), std::move(format), std::move(object), std::move(properties), std::move(kernels));
}

ObjectAttr ObjectAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels) {
  return Base::getChecked(emitError, context, target, format, object, properties, kernels);
}

ObjectAttr ObjectAttr::get(Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels) {
  assert(target && "invalid target");
  return Base::get(target.getContext(), target, format, object, properties, kernels);
}

ObjectAttr ObjectAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels) {
  assert(target && "invalid target");
  return Base::getChecked(emitError, target.getContext(), target, format, object, properties, kernels);
}

::llvm::LogicalResult ObjectAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target, CompilationTarget format, StringAttr object, DictionaryAttr properties, KernelTableAttr kernels) {
  if (::mlir::failed(verify(emitError, target, format, object, properties, kernels)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute ObjectAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Attribute> _result_target;
  ::mlir::FailureOr<CompilationTarget> _result_format;
  ::mlir::FailureOr<StringAttr> _result_object;
  ::mlir::FailureOr<DictionaryAttr> _result_properties;
  ::mlir::FailureOr<KernelTableAttr> _result_kernels;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'target'
  _result_target = ::mlir::FieldParser<Attribute>::parse(odsParser);
  if (::mlir::failed(_result_target)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ObjectAttr parameter 'target' which is to be a `Attribute`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'properties'
  if (odsParser.parseOptionalKeyword("properties")) {
  } else {
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'properties'
    _result_properties = ::mlir::FieldParser<DictionaryAttr>::parse(odsParser);
    if (::mlir::failed(_result_properties)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ObjectAttr parameter 'properties' which is to be a `DictionaryAttr`");
      return {};
    }
    // Parse literal ','
    if (odsParser.parseComma()) return {};
  }
  // Parse literal 'kernels'
  if (odsParser.parseOptionalKeyword("kernels")) {
  } else {
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'kernels'
    _result_kernels = ::mlir::FieldParser<KernelTableAttr>::parse(odsParser);
    if (::mlir::failed(_result_kernels)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ObjectAttr parameter 'kernels' which is to be a `KernelTableAttr`");
      return {};
    }
    // Parse literal ','
    if (odsParser.parseComma()) return {};
  }
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseObject(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_format),
      ::mlir::detail::unwrapForCustomParse(_result_object));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_object)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'object'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_target));
  assert(::mlir::succeeded(_result_object));
  return odsParser.getChecked<ObjectAttr>(odsLoc, odsParser.getContext(),
      Attribute((*_result_target)),
      CompilationTarget((_result_format.value_or(CompilationTarget::Fatbin))),
      StringAttr((*_result_object)),
      DictionaryAttr((_result_properties.value_or(DictionaryAttr()))),
      KernelTableAttr((_result_kernels.value_or(KernelTableAttr()))));
}

void ObjectAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getTarget());
  odsPrinter << ",";
  if (!(getProperties() == DictionaryAttr())) {
    odsPrinter << ' ' << "properties";
    odsPrinter << ' ' << "=";
    if (!(getProperties() == DictionaryAttr())) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getProperties());
    }
    odsPrinter << ",";
  } else {
  }
  if (!(getKernels() == KernelTableAttr())) {
    odsPrinter << ' ' << "kernels";
    odsPrinter << ' ' << "=";
    if (!(getKernels() == KernelTableAttr())) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getKernels());
    }
    odsPrinter << ",";
  } else {
  }
  odsPrinter << ' ';
  printObject(odsPrinter,
    getFormat(),
    getObject());
  odsPrinter << ">";
}

Attribute ObjectAttr::getTarget() const {
  return getImpl()->target;
}

CompilationTarget ObjectAttr::getFormat() const {
  return getImpl()->format;
}

StringAttr ObjectAttr::getObject() const {
  return getImpl()->object;
}

DictionaryAttr ObjectAttr::getProperties() const {
  return getImpl()->properties;
}

KernelTableAttr ObjectAttr::getKernels() const {
  return getImpl()->kernels;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ObjectAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct SelectObjectAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Attribute>;
  SelectObjectAttrStorage(Attribute target) : target(std::move(target)) {}

  KeyTy getAsKey() const {
    return KeyTy(target);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (target == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SelectObjectAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto target = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SelectObjectAttrStorage>()) SelectObjectAttrStorage(std::move(target));
  }

  Attribute target;
};
} // namespace detail
SelectObjectAttr SelectObjectAttr::get(::mlir::MLIRContext *context, Attribute target) {
  return Base::get(context, std::move(target));
}

SelectObjectAttr SelectObjectAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Attribute target) {
  return Base::getChecked(emitError, context, target);
}

::llvm::LogicalResult SelectObjectAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Attribute target) {
  if (::mlir::failed(verify(emitError, target)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute SelectObjectAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Attribute> _result_target;
  // Parse literal '<'
  if (odsParser.parseOptionalLess()) {
  } else {

    // Parse variable 'target'
    _result_target = ::mlir::FieldParser<Attribute>::parse(odsParser);
    if (::mlir::failed(_result_target)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_SelectObjectAttr parameter 'target' which is to be a `Attribute`");
      return {};
    }
    // Parse literal '>'
    if (odsParser.parseGreater()) return {};
  }
  return odsParser.getChecked<SelectObjectAttr>(odsLoc, odsParser.getContext(),
      Attribute((_result_target.value_or(Attribute()))));
}

void SelectObjectAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  if (!(getTarget() == Attribute())) {
    odsPrinter << "<";
    if (!(getTarget() == Attribute())) {
      odsPrinter.printStrippedAttrOrType(getTarget());
    }
    odsPrinter << ">";
  } else {
  }
}

Attribute SelectObjectAttr::getTarget() const {
  return getImpl()->target;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SelectObjectAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUBlockMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MappingId>;
  GPUBlockMappingAttrStorage(::mlir::gpu::MappingId block) : block(std::move(block)) {}

  KeyTy getAsKey() const {
    return KeyTy(block);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (block == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUBlockMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto block = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<GPUBlockMappingAttrStorage>()) GPUBlockMappingAttrStorage(std::move(block));
  }

  ::mlir::gpu::MappingId block;
};
} // namespace detail
GPUBlockMappingAttr GPUBlockMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId block) {
  return Base::get(context, std::move(block));
}

::mlir::Attribute GPUBlockMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::MappingId> _result_block;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'block'
  _result_block = [&]() -> ::mlir::FailureOr<::mlir::gpu::MappingId> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMappingId(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MappingId" << " to be one of: " << "x" << ", " << "y" << ", " << "z" << ", " << "linear_dim_0" << ", " << "linear_dim_1" << ", " << "linear_dim_2" << ", " << "linear_dim_3" << ", " << "linear_dim_4" << ", " << "linear_dim_5" << ", " << "linear_dim_6" << ", " << "linear_dim_7" << ", " << "linear_dim_8" << ", " << "linear_dim_9")};
    }();
  if (::mlir::failed(_result_block)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUBlockMappingAttr parameter 'block' which is to be a `::mlir::gpu::MappingId`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_block));
  return GPUBlockMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::MappingId((*_result_block)));
}

void GPUBlockMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyMappingId(getBlock());
  }
  odsPrinter << ">";
}

::mlir::gpu::MappingId GPUBlockMappingAttr::getBlock() const {
  return getImpl()->block;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUBlockMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUWarpgroupMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MappingId>;
  GPUWarpgroupMappingAttrStorage(::mlir::gpu::MappingId warpgroup) : warpgroup(std::move(warpgroup)) {}

  KeyTy getAsKey() const {
    return KeyTy(warpgroup);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (warpgroup == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUWarpgroupMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto warpgroup = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<GPUWarpgroupMappingAttrStorage>()) GPUWarpgroupMappingAttrStorage(std::move(warpgroup));
  }

  ::mlir::gpu::MappingId warpgroup;
};
} // namespace detail
GPUWarpgroupMappingAttr GPUWarpgroupMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId warpgroup) {
  return Base::get(context, std::move(warpgroup));
}

::mlir::Attribute GPUWarpgroupMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::MappingId> _result_warpgroup;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'warpgroup'
  _result_warpgroup = [&]() -> ::mlir::FailureOr<::mlir::gpu::MappingId> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMappingId(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MappingId" << " to be one of: " << "x" << ", " << "y" << ", " << "z" << ", " << "linear_dim_0" << ", " << "linear_dim_1" << ", " << "linear_dim_2" << ", " << "linear_dim_3" << ", " << "linear_dim_4" << ", " << "linear_dim_5" << ", " << "linear_dim_6" << ", " << "linear_dim_7" << ", " << "linear_dim_8" << ", " << "linear_dim_9")};
    }();
  if (::mlir::failed(_result_warpgroup)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUWarpgroupMappingAttr parameter 'warpgroup' which is to be a `::mlir::gpu::MappingId`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_warpgroup));
  return GPUWarpgroupMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::MappingId((*_result_warpgroup)));
}

void GPUWarpgroupMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyMappingId(getWarpgroup());
  }
  odsPrinter << ">";
}

::mlir::gpu::MappingId GPUWarpgroupMappingAttr::getWarpgroup() const {
  return getImpl()->warpgroup;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUWarpgroupMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUWarpMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MappingId>;
  GPUWarpMappingAttrStorage(::mlir::gpu::MappingId warp) : warp(std::move(warp)) {}

  KeyTy getAsKey() const {
    return KeyTy(warp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (warp == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUWarpMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto warp = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<GPUWarpMappingAttrStorage>()) GPUWarpMappingAttrStorage(std::move(warp));
  }

  ::mlir::gpu::MappingId warp;
};
} // namespace detail
GPUWarpMappingAttr GPUWarpMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId warp) {
  return Base::get(context, std::move(warp));
}

::mlir::Attribute GPUWarpMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::MappingId> _result_warp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'warp'
  _result_warp = [&]() -> ::mlir::FailureOr<::mlir::gpu::MappingId> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMappingId(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MappingId" << " to be one of: " << "x" << ", " << "y" << ", " << "z" << ", " << "linear_dim_0" << ", " << "linear_dim_1" << ", " << "linear_dim_2" << ", " << "linear_dim_3" << ", " << "linear_dim_4" << ", " << "linear_dim_5" << ", " << "linear_dim_6" << ", " << "linear_dim_7" << ", " << "linear_dim_8" << ", " << "linear_dim_9")};
    }();
  if (::mlir::failed(_result_warp)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUWarpMappingAttr parameter 'warp' which is to be a `::mlir::gpu::MappingId`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_warp));
  return GPUWarpMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::MappingId((*_result_warp)));
}

void GPUWarpMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyMappingId(getWarp());
  }
  odsPrinter << ">";
}

::mlir::gpu::MappingId GPUWarpMappingAttr::getWarp() const {
  return getImpl()->warp;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUWarpMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUThreadMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MappingId>;
  GPUThreadMappingAttrStorage(::mlir::gpu::MappingId thread) : thread(std::move(thread)) {}

  KeyTy getAsKey() const {
    return KeyTy(thread);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (thread == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUThreadMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto thread = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<GPUThreadMappingAttrStorage>()) GPUThreadMappingAttrStorage(std::move(thread));
  }

  ::mlir::gpu::MappingId thread;
};
} // namespace detail
GPUThreadMappingAttr GPUThreadMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MappingId thread) {
  return Base::get(context, std::move(thread));
}

::mlir::Attribute GPUThreadMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::MappingId> _result_thread;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'thread'
  _result_thread = [&]() -> ::mlir::FailureOr<::mlir::gpu::MappingId> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMappingId(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MappingId" << " to be one of: " << "x" << ", " << "y" << ", " << "z" << ", " << "linear_dim_0" << ", " << "linear_dim_1" << ", " << "linear_dim_2" << ", " << "linear_dim_3" << ", " << "linear_dim_4" << ", " << "linear_dim_5" << ", " << "linear_dim_6" << ", " << "linear_dim_7" << ", " << "linear_dim_8" << ", " << "linear_dim_9")};
    }();
  if (::mlir::failed(_result_thread)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUThreadMappingAttr parameter 'thread' which is to be a `::mlir::gpu::MappingId`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_thread));
  return GPUThreadMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::MappingId((*_result_thread)));
}

void GPUThreadMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyMappingId(getThread());
  }
  odsPrinter << ">";
}

::mlir::gpu::MappingId GPUThreadMappingAttr::getThread() const {
  return getImpl()->thread;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUThreadMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct GPUMemorySpaceMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AddressSpace>;
  GPUMemorySpaceMappingAttrStorage(::mlir::gpu::AddressSpace address_space) : address_space(std::move(address_space)) {}

  KeyTy getAsKey() const {
    return KeyTy(address_space);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (address_space == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GPUMemorySpaceMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto address_space = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<GPUMemorySpaceMappingAttrStorage>()) GPUMemorySpaceMappingAttrStorage(std::move(address_space));
  }

  ::mlir::gpu::AddressSpace address_space;
};
} // namespace detail
GPUMemorySpaceMappingAttr GPUMemorySpaceMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AddressSpace address_space) {
  return Base::get(context, std::move(address_space));
}

::mlir::Attribute GPUMemorySpaceMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::AddressSpace> _result_address_space;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'address_space'
  _result_address_space = [&]() -> ::mlir::FailureOr<::mlir::gpu::AddressSpace> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAddressSpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AddressSpace" << " to be one of: " << "global" << ", " << "workgroup" << ", " << "private")};
    }();
  if (::mlir::failed(_result_address_space)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPUMemorySpaceMappingAttr parameter 'address_space' which is to be a `::mlir::gpu::AddressSpace`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_address_space));
  return GPUMemorySpaceMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::AddressSpace((*_result_address_space)));
}

void GPUMemorySpaceMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyAddressSpace(getAddressSpace());
  }
  odsPrinter << ">";
}

::mlir::gpu::AddressSpace GPUMemorySpaceMappingAttr::getAddressSpace() const {
  return getImpl()->address_space;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUMemorySpaceMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ParallelLoopDimMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Processor, AffineMap, AffineMap>;
  ParallelLoopDimMappingAttrStorage(::mlir::gpu::Processor processor, AffineMap map, AffineMap bound) : processor(std::move(processor)), map(std::move(map)), bound(std::move(bound)) {}

  KeyTy getAsKey() const {
    return KeyTy(processor, map, bound);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (processor == std::get<0>(tblgenKey)) && (map == std::get<1>(tblgenKey)) && (bound == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static ParallelLoopDimMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto processor = std::move(std::get<0>(tblgenKey));
    auto map = std::move(std::get<1>(tblgenKey));
    auto bound = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<ParallelLoopDimMappingAttrStorage>()) ParallelLoopDimMappingAttrStorage(std::move(processor), std::move(map), std::move(bound));
  }

  ::mlir::gpu::Processor processor;
  AffineMap map;
  AffineMap bound;
};
} // namespace detail
ParallelLoopDimMappingAttr ParallelLoopDimMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Processor processor, AffineMap map, AffineMap bound) {
  return Base::get(context, std::move(processor), std::move(map), std::move(bound));
}

::mlir::Attribute ParallelLoopDimMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Processor> _result_processor;
  ::mlir::FailureOr<AffineMap> _result_map;
  ::mlir::FailureOr<AffineMap> _result_bound;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_processor = false;
  bool _seen_map = false;
  bool _seen_bound = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_processor && _paramKey == "processor") {
        _seen_processor = true;

        // Parse variable 'processor'
        _result_processor = [&]() -> ::mlir::FailureOr<::mlir::gpu::Processor> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::gpu::symbolizeProcessor(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Processor" << " to be one of: " << "block_x" << ", " << "block_y" << ", " << "block_z" << ", " << "thread_x" << ", " << "thread_y" << ", " << "thread_z" << ", " << "sequential")};
          }();
        if (::mlir::failed(_result_processor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'processor' which is to be a `::mlir::gpu::Processor`");
          return {};
        }
      } else if (!_seen_map && _paramKey == "map") {
        _seen_map = true;

        // Parse variable 'map'
        _result_map = ::mlir::FieldParser<AffineMap>::parse(odsParser);
        if (::mlir::failed(_result_map)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'map' which is to be a `AffineMap`");
          return {};
        }
      } else if (!_seen_bound && _paramKey == "bound") {
        _seen_bound = true;

        // Parse variable 'bound'
        _result_bound = ::mlir::FieldParser<AffineMap>::parse(odsParser);
        if (::mlir::failed(_result_bound)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'bound' which is to be a `AffineMap`");
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
  assert(::mlir::succeeded(_result_processor));
  assert(::mlir::succeeded(_result_map));
  assert(::mlir::succeeded(_result_bound));
  return ParallelLoopDimMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::Processor((*_result_processor)),
      AffineMap((*_result_map)),
      AffineMap((*_result_bound)));
}

void ParallelLoopDimMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "processor = ";
    odsPrinter << stringifyProcessor(getProcessor());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "map = ";
    odsPrinter.printStrippedAttrOrType(getMap());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "bound = ";
    odsPrinter.printStrippedAttrOrType(getBound());
  }
  odsPrinter << ">";
}

::mlir::gpu::Processor ParallelLoopDimMappingAttr::getProcessor() const {
  return getImpl()->processor;
}

AffineMap ParallelLoopDimMappingAttr::getMap() const {
  return getImpl()->map;
}

AffineMap ParallelLoopDimMappingAttr::getBound() const {
  return getImpl()->bound;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ParallelLoopDimMappingAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct DimensionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Dimension>;
  DimensionAttrStorage(::mlir::gpu::Dimension value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DimensionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DimensionAttrStorage>()) DimensionAttrStorage(std::move(value));
  }

  ::mlir::gpu::Dimension value;
};
} // namespace detail
DimensionAttr DimensionAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Dimension value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DimensionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Dimension> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::Dimension> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeDimension(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Dimension" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_DimensionAttr parameter 'value' which is to be a `::mlir::gpu::Dimension`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return DimensionAttr::get(odsParser.getContext(),
      ::mlir::gpu::Dimension((*_result_value)));
}

void DimensionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyDimension(getValue());
}

::mlir::gpu::Dimension DimensionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct AllReduceOperationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AllReduceOperation>;
  AllReduceOperationAttrStorage(::mlir::gpu::AllReduceOperation value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AllReduceOperationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<AllReduceOperationAttrStorage>()) AllReduceOperationAttrStorage(std::move(value));
  }

  ::mlir::gpu::AllReduceOperation value;
};
} // namespace detail
AllReduceOperationAttr AllReduceOperationAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AllReduceOperation value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute AllReduceOperationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::AllReduceOperation> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::AllReduceOperation> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAllReduceOperation(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AllReduceOperation" << " to be one of: " << "add" << ", " << "mul" << ", " << "minui" << ", " << "minsi" << ", " << "minnumf" << ", " << "maxui" << ", " << "maxsi" << ", " << "maxnumf" << ", " << "and" << ", " << "or" << ", " << "xor" << ", " << "minimumf" << ", " << "maximumf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_AllReduceOperationAttr parameter 'value' which is to be a `::mlir::gpu::AllReduceOperation`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return AllReduceOperationAttr::get(odsParser.getContext(),
      ::mlir::gpu::AllReduceOperation((*_result_value)));
}

void AllReduceOperationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyAllReduceOperation(getValue());
}

::mlir::gpu::AllReduceOperation AllReduceOperationAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ShuffleModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::ShuffleMode>;
  ShuffleModeAttrStorage(::mlir::gpu::ShuffleMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ShuffleModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ShuffleModeAttrStorage>()) ShuffleModeAttrStorage(std::move(value));
  }

  ::mlir::gpu::ShuffleMode value;
};
} // namespace detail
ShuffleModeAttr ShuffleModeAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::ShuffleMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ShuffleModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::ShuffleMode> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::ShuffleMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeShuffleMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::ShuffleMode" << " to be one of: " << "xor" << ", " << "up" << ", " << "down" << ", " << "idx")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ShuffleModeAttr parameter 'value' which is to be a `::mlir::gpu::ShuffleMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ShuffleModeAttr::get(odsParser.getContext(),
      ::mlir::gpu::ShuffleMode((*_result_value)));
}

void ShuffleModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyShuffleMode(getValue());
}

::mlir::gpu::ShuffleMode ShuffleModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct MMAElementwiseOpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MMAElementwiseOp>;
  MMAElementwiseOpAttrStorage(::mlir::gpu::MMAElementwiseOp value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMAElementwiseOpAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<MMAElementwiseOpAttrStorage>()) MMAElementwiseOpAttrStorage(std::move(value));
  }

  ::mlir::gpu::MMAElementwiseOp value;
};
} // namespace detail
MMAElementwiseOpAttr MMAElementwiseOpAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MMAElementwiseOp value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute MMAElementwiseOpAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::MMAElementwiseOp> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::MMAElementwiseOp> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMMAElementwiseOp(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MMAElementwiseOp" << " to be one of: " << "addf" << ", " << "mulf" << ", " << "subf" << ", " << "maxf" << ", " << "minf" << ", " << "divf" << ", " << "addi" << ", " << "muli" << ", " << "subi" << ", " << "divs" << ", " << "divu" << ", " << "negatef" << ", " << "negates" << ", " << "extf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAElementWiseAttr parameter 'value' which is to be a `::mlir::gpu::MMAElementwiseOp`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MMAElementwiseOpAttr::get(odsParser.getContext(),
      ::mlir::gpu::MMAElementwiseOp((*_result_value)));
}

void MMAElementwiseOpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyMMAElementwiseOp(getValue());
}

::mlir::gpu::MMAElementwiseOp MMAElementwiseOpAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct Prune2To4SpMatFlagAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Prune2To4SpMatFlag>;
  Prune2To4SpMatFlagAttrStorage(::mlir::gpu::Prune2To4SpMatFlag value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static Prune2To4SpMatFlagAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<Prune2To4SpMatFlagAttrStorage>()) Prune2To4SpMatFlagAttrStorage(std::move(value));
  }

  ::mlir::gpu::Prune2To4SpMatFlag value;
};
} // namespace detail
Prune2To4SpMatFlagAttr Prune2To4SpMatFlagAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Prune2To4SpMatFlag value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute Prune2To4SpMatFlagAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::Prune2To4SpMatFlag> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::Prune2To4SpMatFlag> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizePrune2To4SpMatFlag(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Prune2To4SpMatFlag" << " to be one of: " << "NONE" << ", " << "PRUNE_ONLY" << ", " << "PRUNE_AND_CHECK")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_Prune2To4SpMatFlagAttr parameter 'value' which is to be a `::mlir::gpu::Prune2To4SpMatFlag`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return Prune2To4SpMatFlagAttr::get(odsParser.getContext(),
      ::mlir::gpu::Prune2To4SpMatFlag((*_result_value)));
}

void Prune2To4SpMatFlagAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyPrune2To4SpMatFlag(getValue());
}

::mlir::gpu::Prune2To4SpMatFlag Prune2To4SpMatFlagAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::Prune2To4SpMatFlagAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct TransposeModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::TransposeMode>;
  TransposeModeAttrStorage(::mlir::gpu::TransposeMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TransposeModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TransposeModeAttrStorage>()) TransposeModeAttrStorage(std::move(value));
  }

  ::mlir::gpu::TransposeMode value;
};
} // namespace detail
TransposeModeAttr TransposeModeAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::TransposeMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TransposeModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::TransposeMode> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::TransposeMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeTransposeMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::TransposeMode" << " to be one of: " << "NON_TRANSPOSE" << ", " << "TRANSPOSE" << ", " << "CONJUGATE_TRANSPOSE")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_TransposeModeAttr parameter 'value' which is to be a `::mlir::gpu::TransposeMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TransposeModeAttr::get(odsParser.getContext(),
      ::mlir::gpu::TransposeMode((*_result_value)));
}

void TransposeModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTransposeMode(getValue());
}

::mlir::gpu::TransposeMode TransposeModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::TransposeModeAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct SpGEMMWorkEstimationOrComputeKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind>;
  SpGEMMWorkEstimationOrComputeKindAttrStorage(::mlir::gpu::SpGEMMWorkEstimationOrComputeKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SpGEMMWorkEstimationOrComputeKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SpGEMMWorkEstimationOrComputeKindAttrStorage>()) SpGEMMWorkEstimationOrComputeKindAttrStorage(std::move(value));
  }

  ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind value;
};
} // namespace detail
SpGEMMWorkEstimationOrComputeKindAttr SpGEMMWorkEstimationOrComputeKindAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute SpGEMMWorkEstimationOrComputeKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::SpGEMMWorkEstimationOrComputeKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeSpGEMMWorkEstimationOrComputeKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::SpGEMMWorkEstimationOrComputeKind" << " to be one of: " << "WORK_ESTIMATION" << ", " << "COMPUTE")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_SpGEMMWorkEstimationOrComputeKindAttr parameter 'value' which is to be a `::mlir::gpu::SpGEMMWorkEstimationOrComputeKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return SpGEMMWorkEstimationOrComputeKindAttr::get(odsParser.getContext(),
      ::mlir::gpu::SpGEMMWorkEstimationOrComputeKind((*_result_value)));
}

void SpGEMMWorkEstimationOrComputeKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifySpGEMMWorkEstimationOrComputeKind(getValue());
}

::mlir::gpu::SpGEMMWorkEstimationOrComputeKind SpGEMMWorkEstimationOrComputeKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr)
namespace mlir {
namespace gpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute GPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void GPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace gpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

