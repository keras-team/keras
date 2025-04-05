/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::sparse_tensor::SparseTensorDimSliceAttr,
::mlir::sparse_tensor::SparseTensorEncodingAttr,
::mlir::sparse_tensor::StorageSpecifierKindAttr,
::mlir::sparse_tensor::SparseTensorSortKindAttr,
::mlir::sparse_tensor::CrdTransDirectionKindAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::sparse_tensor::SparseTensorDimSliceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::SparseTensorDimSliceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::SparseTensorEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::StorageSpecifierKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::StorageSpecifierKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::SparseTensorSortKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::SparseTensorSortKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::CrdTransDirectionKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::CrdTransDirectionKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::sparse_tensor::SparseTensorDimSliceAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorDimSliceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::SparseTensorEncodingAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::StorageSpecifierKindAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::StorageSpecifierKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::SparseTensorSortKindAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorSortKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::CrdTransDirectionKindAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::CrdTransDirectionKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorDimSliceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t, int64_t>;
  SparseTensorDimSliceAttrStorage(int64_t offset, int64_t size, int64_t stride) : offset(std::move(offset)), size(std::move(size)), stride(std::move(stride)) {}

  KeyTy getAsKey() const {
    return KeyTy(offset, size, stride);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (offset == std::get<0>(tblgenKey)) && (size == std::get<1>(tblgenKey)) && (stride == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static SparseTensorDimSliceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto offset = std::move(std::get<0>(tblgenKey));
    auto size = std::move(std::get<1>(tblgenKey));
    auto stride = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<SparseTensorDimSliceAttrStorage>()) SparseTensorDimSliceAttrStorage(std::move(offset), std::move(size), std::move(stride));
  }

  int64_t offset;
  int64_t size;
  int64_t stride;
};
} // namespace detail
SparseTensorDimSliceAttr SparseTensorDimSliceAttr::get(::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride) {
  return Base::get(context, std::move(offset), std::move(size), std::move(stride));
}

SparseTensorDimSliceAttr SparseTensorDimSliceAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride) {
  return Base::getChecked(emitError, context, offset, size, stride);
}

SparseTensorDimSliceAttr SparseTensorDimSliceAttr::get(::mlir::MLIRContext *context) {
  return Base::get(context, 0, kDynamic, 1); 
}

SparseTensorDimSliceAttr SparseTensorDimSliceAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context) {
  return Base::getChecked(emitError, context, 0, kDynamic, 1); 
}

::llvm::LogicalResult SparseTensorDimSliceAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t offset, int64_t size, int64_t stride) {
  if (::mlir::failed(verify(emitError, offset, size, stride)))
    return ::mlir::failure();
  return ::mlir::success();
}

int64_t SparseTensorDimSliceAttr::getOffset() const {
  return getImpl()->offset;
}

int64_t SparseTensorDimSliceAttr::getSize() const {
  return getImpl()->size;
}

int64_t SparseTensorDimSliceAttr::getStride() const {
  return getImpl()->stride;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorDimSliceAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<::mlir::sparse_tensor::LevelType>, AffineMap, AffineMap, unsigned, unsigned, ::mlir::Attribute, ::mlir::Attribute, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>>;
  SparseTensorEncodingAttrStorage(::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) : lvlTypes(std::move(lvlTypes)), dimToLvl(std::move(dimToLvl)), lvlToDim(std::move(lvlToDim)), posWidth(std::move(posWidth)), crdWidth(std::move(crdWidth)), explicitVal(std::move(explicitVal)), implicitVal(std::move(implicitVal)), dimSlices(std::move(dimSlices)) {}

  KeyTy getAsKey() const {
    return KeyTy(lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth, explicitVal, implicitVal, dimSlices);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (lvlTypes == std::get<0>(tblgenKey)) && (dimToLvl == std::get<1>(tblgenKey)) && (lvlToDim == std::get<2>(tblgenKey)) && (posWidth == std::get<3>(tblgenKey)) && (crdWidth == std::get<4>(tblgenKey)) && (explicitVal == std::get<5>(tblgenKey)) && (implicitVal == std::get<6>(tblgenKey)) && (dimSlices == std::get<7>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey), std::get<7>(tblgenKey));
  }

  static SparseTensorEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto lvlTypes = std::move(std::get<0>(tblgenKey));
    auto dimToLvl = std::move(std::get<1>(tblgenKey));
    auto lvlToDim = std::move(std::get<2>(tblgenKey));
    auto posWidth = std::move(std::get<3>(tblgenKey));
    auto crdWidth = std::move(std::get<4>(tblgenKey));
    auto explicitVal = std::move(std::get<5>(tblgenKey));
    auto implicitVal = std::move(std::get<6>(tblgenKey));
    auto dimSlices = std::move(std::get<7>(tblgenKey));
    lvlTypes = allocator.copyInto(lvlTypes);
    dimSlices = allocator.copyInto(dimSlices);
    return new (allocator.allocate<SparseTensorEncodingAttrStorage>()) SparseTensorEncodingAttrStorage(std::move(lvlTypes), std::move(dimToLvl), std::move(lvlToDim), std::move(posWidth), std::move(crdWidth), std::move(explicitVal), std::move(implicitVal), std::move(dimSlices));
  }

  ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes;
  AffineMap dimToLvl;
  AffineMap lvlToDim;
  unsigned posWidth;
  unsigned crdWidth;
  ::mlir::Attribute explicitVal;
  ::mlir::Attribute implicitVal;
  ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices;
};
} // namespace detail
SparseTensorEncodingAttr SparseTensorEncodingAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) {
  return Base::get(context, std::move(lvlTypes), std::move(dimToLvl), std::move(lvlToDim), std::move(posWidth), std::move(crdWidth), std::move(explicitVal), std::move(implicitVal), std::move(dimSlices));
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) {
  return Base::getChecked(emitError, context, lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth, explicitVal, implicitVal, dimSlices);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::get(::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal) {
  if (!dimToLvl) {
    dimToLvl = ::mlir::AffineMap::getMultiDimIdentityMap(lvlTypes.size(), context);
  }
  if (!lvlToDim) {
    lvlToDim = ::mlir::sparse_tensor::inferLvlToDim(dimToLvl, context);
  }
  return Base::get(context, lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth,
    explicitVal, implicitVal,
    ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>{});
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal) {
  if (!dimToLvl) {
    dimToLvl = ::mlir::AffineMap::getMultiDimIdentityMap(lvlTypes.size(), context);
  }
  if (!lvlToDim) {
    lvlToDim = ::mlir::sparse_tensor::inferLvlToDim(dimToLvl, context);
  }
  return Base::getChecked(emitError, context, lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth,
    explicitVal, implicitVal,
    ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>{});
}

::llvm::LogicalResult SparseTensorEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> lvlTypes, AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth, unsigned crdWidth, ::mlir::Attribute explicitVal, ::mlir::Attribute implicitVal, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) {
  if (::mlir::failed(verify(emitError, lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth, explicitVal, implicitVal, dimSlices)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<::mlir::sparse_tensor::LevelType> SparseTensorEncodingAttr::getLvlTypes() const {
  return getImpl()->lvlTypes;
}

AffineMap SparseTensorEncodingAttr::getDimToLvl() const {
  return getImpl()->dimToLvl;
}

AffineMap SparseTensorEncodingAttr::getLvlToDim() const {
  return getImpl()->lvlToDim;
}

unsigned SparseTensorEncodingAttr::getPosWidth() const {
  return getImpl()->posWidth;
}

unsigned SparseTensorEncodingAttr::getCrdWidth() const {
  return getImpl()->crdWidth;
}

::mlir::Attribute SparseTensorEncodingAttr::getExplicitVal() const {
  return getImpl()->explicitVal;
}

::mlir::Attribute SparseTensorEncodingAttr::getImplicitVal() const {
  return getImpl()->implicitVal;
}

::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> SparseTensorEncodingAttr::getDimSlices() const {
  return getImpl()->dimSlices;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct StorageSpecifierKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::StorageSpecifierKind>;
  StorageSpecifierKindAttrStorage(::mlir::sparse_tensor::StorageSpecifierKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static StorageSpecifierKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<StorageSpecifierKindAttrStorage>()) StorageSpecifierKindAttrStorage(std::move(value));
  }

  ::mlir::sparse_tensor::StorageSpecifierKind value;
};
} // namespace detail
StorageSpecifierKindAttr StorageSpecifierKindAttr::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::StorageSpecifierKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute StorageSpecifierKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::StorageSpecifierKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::sparse_tensor::StorageSpecifierKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::sparse_tensor::symbolizeStorageSpecifierKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::sparse_tensor::StorageSpecifierKind" << " to be one of: " << "lvl_sz" << ", " << "pos_mem_sz" << ", " << "crd_mem_sz" << ", " << "val_mem_sz" << ", " << "dim_offset" << ", " << "dim_stride")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensorStorageSpecifierKindAttr parameter 'value' which is to be a `::mlir::sparse_tensor::StorageSpecifierKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return StorageSpecifierKindAttr::get(odsParser.getContext(),
      ::mlir::sparse_tensor::StorageSpecifierKind((*_result_value)));
}

void StorageSpecifierKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyStorageSpecifierKind(getValue());
}

::mlir::sparse_tensor::StorageSpecifierKind StorageSpecifierKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierKindAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorSortKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::SparseTensorSortKind>;
  SparseTensorSortKindAttrStorage(::mlir::sparse_tensor::SparseTensorSortKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SparseTensorSortKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SparseTensorSortKindAttrStorage>()) SparseTensorSortKindAttrStorage(std::move(value));
  }

  ::mlir::sparse_tensor::SparseTensorSortKind value;
};
} // namespace detail
SparseTensorSortKindAttr SparseTensorSortKindAttr::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorSortKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute SparseTensorSortKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorSortKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorSortKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::sparse_tensor::symbolizeSparseTensorSortKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::sparse_tensor::SparseTensorSortKind" << " to be one of: " << "hybrid_quick_sort" << ", " << "insertion_sort_stable" << ", " << "quick_sort" << ", " << "heap_sort")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensorSortKindAttr parameter 'value' which is to be a `::mlir::sparse_tensor::SparseTensorSortKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return SparseTensorSortKindAttr::get(odsParser.getContext(),
      ::mlir::sparse_tensor::SparseTensorSortKind((*_result_value)));
}

void SparseTensorSortKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifySparseTensorSortKind(getValue());
}

::mlir::sparse_tensor::SparseTensorSortKind SparseTensorSortKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorSortKindAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct CrdTransDirectionKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::CrdTransDirectionKind>;
  CrdTransDirectionKindAttrStorage(::mlir::sparse_tensor::CrdTransDirectionKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CrdTransDirectionKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CrdTransDirectionKindAttrStorage>()) CrdTransDirectionKindAttrStorage(std::move(value));
  }

  ::mlir::sparse_tensor::CrdTransDirectionKind value;
};
} // namespace detail
CrdTransDirectionKindAttr CrdTransDirectionKindAttr::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::CrdTransDirectionKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CrdTransDirectionKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::CrdTransDirectionKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::sparse_tensor::CrdTransDirectionKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::sparse_tensor::symbolizeCrdTransDirectionKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::sparse_tensor::CrdTransDirectionKind" << " to be one of: " << "dim_to_lvl" << ", " << "lvl_to_dim")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensorCrdTransDirectionAttr parameter 'value' which is to be a `::mlir::sparse_tensor::CrdTransDirectionKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return CrdTransDirectionKindAttr::get(odsParser.getContext(),
      ::mlir::sparse_tensor::CrdTransDirectionKind((*_result_value)));
}

void CrdTransDirectionKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyCrdTransDirectionKind(getValue());
}

::mlir::sparse_tensor::CrdTransDirectionKind CrdTransDirectionKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::CrdTransDirectionKindAttr)
namespace mlir {
namespace sparse_tensor {

/// Parse an attribute registered to this dialect.
::mlir::Attribute SparseTensorDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void SparseTensorDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace sparse_tensor
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

