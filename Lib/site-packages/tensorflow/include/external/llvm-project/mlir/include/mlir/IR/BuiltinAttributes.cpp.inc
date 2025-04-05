/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::AffineMapAttr,
::mlir::ArrayAttr,
::mlir::DenseArrayAttr,
::mlir::DenseIntOrFPElementsAttr,
::mlir::DenseStringElementsAttr,
::mlir::DenseResourceElementsAttr,
::mlir::DictionaryAttr,
::mlir::FloatAttr,
::mlir::IntegerAttr,
::mlir::IntegerSetAttr,
::mlir::OpaqueAttr,
::mlir::SparseElementsAttr,
::mlir::StridedLayoutAttr,
::mlir::StringAttr,
::mlir::SymbolRefAttr,
::mlir::TypeAttr,
::mlir::UnitAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

namespace mlir {
namespace detail {
struct AffineMapAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<AffineMap>;
  AffineMapAttrStorage(AffineMap value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AffineMapAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<AffineMapAttrStorage>()) AffineMapAttrStorage(std::move(value));
  }

  AffineMap value;
};
} // namespace detail
AffineMapAttr AffineMapAttr::get(AffineMap value) {
  return Base::get(value.getContext(), value);
}

AffineMap AffineMapAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::AffineMapAttr)
namespace mlir {
namespace detail {
struct ArrayAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<Attribute>>;
  ArrayAttrStorage(::llvm::ArrayRef<Attribute> value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<ArrayAttrStorage>()) ArrayAttrStorage(std::move(value));
  }

  ::llvm::ArrayRef<Attribute> value;
};
} // namespace detail
ArrayAttr ArrayAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Attribute> value) {
  return Base::get(context, std::move(value));
}

::llvm::ArrayRef<Attribute> ArrayAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ArrayAttr)
namespace mlir {
namespace detail {
struct DenseArrayAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Type, int64_t, ::llvm::ArrayRef<char>>;
  DenseArrayAttrStorage(Type elementType, int64_t size, ::llvm::ArrayRef<char> rawData) : elementType(std::move(elementType)), size(std::move(size)), rawData(std::move(rawData)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, size, rawData);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (size == std::get<1>(tblgenKey)) && (rawData == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static DenseArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    auto size = std::move(std::get<1>(tblgenKey));
    auto rawData = std::move(std::get<2>(tblgenKey));

        if (!rawData.empty()) {
          auto *alloc = static_cast<char *>(
              allocator.allocate(rawData.size(), alignof(uint64_t)));
          std::uninitialized_copy(rawData.begin(), rawData.end(), alloc);
          rawData = ArrayRef<char>(alloc, rawData.size());
        }

    return new (allocator.allocate<DenseArrayAttrStorage>()) DenseArrayAttrStorage(std::move(elementType), std::move(size), std::move(rawData));
  }

  Type elementType;
  int64_t size;
  ::llvm::ArrayRef<char> rawData;
};
} // namespace detail
DenseArrayAttr DenseArrayAttr::get(::mlir::MLIRContext *context, Type elementType, int64_t size, ::llvm::ArrayRef<char> rawData) {
  return Base::get(context, std::move(elementType), std::move(size), std::move(rawData));
}

DenseArrayAttr DenseArrayAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, int64_t size, ::llvm::ArrayRef<char> rawData) {
  return Base::getChecked(emitError, context, elementType, size, rawData);
}

DenseArrayAttr DenseArrayAttr::get(Type elementType, unsigned size, ArrayRef<char> rawData) {
  return Base::get(elementType.getContext(), elementType, size, rawData);
}

DenseArrayAttr DenseArrayAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned size, ArrayRef<char> rawData) {
  return Base::getChecked(emitError, elementType.getContext(), elementType, size, rawData);
}

::llvm::LogicalResult DenseArrayAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, int64_t size, ::llvm::ArrayRef<char> rawData) {
  if (::mlir::failed(verify(emitError, elementType, size, rawData)))
    return ::mlir::failure();
  return ::mlir::success();
}

Type DenseArrayAttr::getElementType() const {
  return getImpl()->elementType;
}

int64_t DenseArrayAttr::getSize() const {
  return getImpl()->size;
}

::llvm::ArrayRef<char> DenseArrayAttr::getRawData() const {
  return getImpl()->rawData;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DenseArrayAttr)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DenseIntOrFPElementsAttr)
namespace mlir {
DenseStringElementsAttr DenseStringElementsAttr::get(ShapedType type, ArrayRef<StringRef> values) {
  return Base::get(type.getContext(), type, values,
               /* isSplat */(values.size() == 1));
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DenseStringElementsAttr)
namespace mlir {
namespace detail {
struct DenseResourceElementsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<ShapedType, DenseResourceElementsHandle>;
  DenseResourceElementsAttrStorage(ShapedType type, DenseResourceElementsHandle rawHandle) : type(std::move(type)), rawHandle(std::move(rawHandle)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, rawHandle);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (rawHandle == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static DenseResourceElementsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto rawHandle = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<DenseResourceElementsAttrStorage>()) DenseResourceElementsAttrStorage(std::move(type), std::move(rawHandle));
  }

  ShapedType type;
  DenseResourceElementsHandle rawHandle;
};
} // namespace detail
ShapedType DenseResourceElementsAttr::getType() const {
  return getImpl()->type;
}

DenseResourceElementsHandle DenseResourceElementsAttr::getRawHandle() const {
  return getImpl()->rawHandle;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DenseResourceElementsAttr)
namespace mlir {
namespace detail {
struct DictionaryAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<NamedAttribute>>;
  DictionaryAttrStorage(::llvm::ArrayRef<NamedAttribute> value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DictionaryAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<DictionaryAttrStorage>()) DictionaryAttrStorage(std::move(value));
  }

  ::llvm::ArrayRef<NamedAttribute> value;
};
} // namespace detail
::llvm::ArrayRef<NamedAttribute> DictionaryAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DictionaryAttr)
namespace mlir {
namespace detail {
struct FloatAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type, ::llvm::APFloat>;
  FloatAttrStorage(::mlir::Type type, ::llvm::APFloat value) : type(std::move(type)), value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (value.bitwiseIsEqual(std::get<1>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FloatAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto value = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<FloatAttrStorage>()) FloatAttrStorage(std::move(type), std::move(value));
  }

  ::mlir::Type type;
  ::llvm::APFloat value;
};
} // namespace detail
FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  return Base::get(type.getContext(), type, value);
}

FloatAttr FloatAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, const APFloat &value) {
  return Base::getChecked(emitError, type.getContext(), type, value);
}

FloatAttr FloatAttr::get(Type type, double value) {
  if (type.isF64() || !::llvm::isa<FloatType>(type))
    return Base::get(type.getContext(), type, APFloat(value));

  // This handles, e.g., F16 because there is no APFloat constructor for it.
  bool unused;
  APFloat val(value);
  val.convert(::llvm::cast<FloatType>(type).getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &unused);
  return Base::get(type.getContext(), type, val);
}

FloatAttr FloatAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, double value) {
  if (type.isF64() || !::llvm::isa<FloatType>(type))
    return Base::getChecked(emitError, type.getContext(), type, APFloat(value));

  // This handles, e.g., F16 because there is no APFloat constructor for it.
  bool unused;
  APFloat val(value);
  val.convert(::llvm::cast<FloatType>(type).getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &unused);
  return Base::getChecked(emitError, type.getContext(), type, val);
}

::llvm::LogicalResult FloatAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, ::llvm::APFloat value) {
  if (::mlir::failed(verify(emitError, type, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type FloatAttr::getType() const {
  return getImpl()->type;
}

::llvm::APFloat FloatAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::FloatAttr)
namespace mlir {
namespace detail {
struct IntegerAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type, APInt>;
  IntegerAttrStorage(::mlir::Type type, APInt value) : type(std::move(type)), value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (value == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static IntegerAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto value = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<IntegerAttrStorage>()) IntegerAttrStorage(std::move(type), std::move(value));
  }

  ::mlir::Type type;
  APInt value;
};
} // namespace detail
IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  if (type.isSignlessInteger(1))
    return BoolAttr::get(type.getContext(), value.getBoolValue());
  return Base::get(type.getContext(), type, value);
}

IntegerAttr IntegerAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, const APInt &value) {
  if (type.isSignlessInteger(1))
    return BoolAttr::get(type.getContext(), value.getBoolValue());
  return Base::getChecked(emitError, type.getContext(), type, value);
}

IntegerAttr IntegerAttr::get(::mlir::MLIRContext *context, const APSInt &value) {
  auto signedness = value.isSigned() ?
    IntegerType::Signed : IntegerType::Unsigned;
  auto type = IntegerType::get(context, value.getBitWidth(), signedness);
  return Base::get(type.getContext(), type, value);
}

IntegerAttr IntegerAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, const APSInt &value) {
  auto signedness = value.isSigned() ?
    IntegerType::Signed : IntegerType::Unsigned;
  auto type = IntegerType::get(context, value.getBitWidth(), signedness);
  return Base::getChecked(emitError, type.getContext(), type, value);
}

IntegerAttr IntegerAttr::get(Type type, int64_t value) {
  // `index` has a defined internal storage width.
  if (type.isIndex()) {
    APInt apValue(IndexType::kInternalStorageBitWidth, value);
    return Base::get(type.getContext(), type, apValue);
  }

  IntegerType intTy = ::llvm::cast<IntegerType>(type);
  APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
  return Base::get(type.getContext(), type, apValue);
}

IntegerAttr IntegerAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, int64_t value) {
  // `index` has a defined internal storage width.
  if (type.isIndex()) {
    APInt apValue(IndexType::kInternalStorageBitWidth, value);
    return Base::getChecked(emitError, type.getContext(), type, apValue);
  }

  IntegerType intTy = ::llvm::cast<IntegerType>(type);
  APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
  return Base::getChecked(emitError, type.getContext(), type, apValue);
}

::llvm::LogicalResult IntegerAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type, APInt value) {
  if (::mlir::failed(verify(emitError, type, value)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type IntegerAttr::getType() const {
  return getImpl()->type;
}

APInt IntegerAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerAttr)
namespace mlir {
namespace detail {
struct IntegerSetAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<IntegerSet>;
  IntegerSetAttrStorage(IntegerSet value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IntegerSetAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<IntegerSetAttrStorage>()) IntegerSetAttrStorage(std::move(value));
  }

  IntegerSet value;
};
} // namespace detail
IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  return Base::get(value.getContext(), value);
}

IntegerSet IntegerSetAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerSetAttr)
namespace mlir {
namespace detail {
struct OpaqueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::StringRef, ::mlir::Type>;
  OpaqueAttrStorage(StringAttr dialectNamespace, ::llvm::StringRef attrData, ::mlir::Type type) : dialectNamespace(std::move(dialectNamespace)), attrData(std::move(attrData)), type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(dialectNamespace, attrData, type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (dialectNamespace == std::get<0>(tblgenKey)) && (attrData == std::get<1>(tblgenKey)) && (type == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static OpaqueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto dialectNamespace = std::move(std::get<0>(tblgenKey));
    auto attrData = std::move(std::get<1>(tblgenKey));
    auto type = std::move(std::get<2>(tblgenKey));
    attrData = allocator.copyInto(attrData);
    return new (allocator.allocate<OpaqueAttrStorage>()) OpaqueAttrStorage(std::move(dialectNamespace), std::move(attrData), std::move(type));
  }

  StringAttr dialectNamespace;
  ::llvm::StringRef attrData;
  ::mlir::Type type;
};
} // namespace detail
OpaqueAttr OpaqueAttr::get(StringAttr dialect, StringRef attrData, Type type) {
  return Base::get(dialect.getContext(), dialect, attrData, type);
}

OpaqueAttr OpaqueAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialect, StringRef attrData, Type type) {
  return Base::getChecked(emitError, dialect.getContext(), dialect, attrData, type);
}

::llvm::LogicalResult OpaqueAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, ::llvm::StringRef attrData, ::mlir::Type type) {
  if (::mlir::failed(verify(emitError, dialectNamespace, attrData, type)))
    return ::mlir::failure();
  return ::mlir::success();
}

StringAttr OpaqueAttr::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

::llvm::StringRef OpaqueAttr::getAttrData() const {
  return getImpl()->attrData;
}

::mlir::Type OpaqueAttr::getType() const {
  return getImpl()->type;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::OpaqueAttr)
namespace mlir {
namespace detail {
struct SparseElementsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<ShapedType, DenseIntElementsAttr, DenseElementsAttr>;
  SparseElementsAttrStorage(ShapedType type, DenseIntElementsAttr indices, DenseElementsAttr values) : type(std::move(type)), indices(std::move(indices)), values(std::move(values)) {}

  KeyTy getAsKey() const {
    return KeyTy(type, indices, values);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (indices == std::get<1>(tblgenKey)) && (values == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static SparseElementsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    auto indices = std::move(std::get<1>(tblgenKey));
    auto values = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<SparseElementsAttrStorage>()) SparseElementsAttrStorage(std::move(type), std::move(indices), std::move(values));
  }

  ShapedType type;
  DenseIntElementsAttr indices;
  DenseElementsAttr values;
};
} // namespace detail
SparseElementsAttr SparseElementsAttr::get(ShapedType type, DenseElementsAttr indices, DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  assert((::llvm::isa<RankedTensorType, VectorType>(type)) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::get(type.getContext(), type,
               ::llvm::cast<DenseIntElementsAttr>(indices), values);
}

SparseElementsAttr SparseElementsAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ShapedType type, DenseElementsAttr indices, DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  assert((::llvm::isa<RankedTensorType, VectorType>(type)) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::getChecked(emitError, type.getContext(), type,
               ::llvm::cast<DenseIntElementsAttr>(indices), values);
}

::llvm::LogicalResult SparseElementsAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ShapedType type, DenseIntElementsAttr indices, DenseElementsAttr values) {
  if (::mlir::failed(verify(emitError, type, indices, values)))
    return ::mlir::failure();
  return ::mlir::success();
}

ShapedType SparseElementsAttr::getType() const {
  return getImpl()->type;
}

DenseIntElementsAttr SparseElementsAttr::getIndices() const {
  return getImpl()->indices;
}

DenseElementsAttr SparseElementsAttr::getValues() const {
  return getImpl()->values;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::SparseElementsAttr)
namespace mlir {
namespace detail {
struct StridedLayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, ::llvm::ArrayRef<int64_t>>;
  StridedLayoutAttrStorage(int64_t offset, ::llvm::ArrayRef<int64_t> strides) : offset(std::move(offset)), strides(std::move(strides)) {}

  KeyTy getAsKey() const {
    return KeyTy(offset, strides);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (offset == std::get<0>(tblgenKey)) && (strides == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static StridedLayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto offset = std::move(std::get<0>(tblgenKey));
    auto strides = std::move(std::get<1>(tblgenKey));
    strides = allocator.copyInto(strides);
    return new (allocator.allocate<StridedLayoutAttrStorage>()) StridedLayoutAttrStorage(std::move(offset), std::move(strides));
  }

  int64_t offset;
  ::llvm::ArrayRef<int64_t> strides;
};
} // namespace detail
StridedLayoutAttr StridedLayoutAttr::get(::mlir::MLIRContext *context, int64_t offset, ::llvm::ArrayRef<int64_t> strides) {
  return Base::get(context, std::move(offset), std::move(strides));
}

StridedLayoutAttr StridedLayoutAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int64_t offset, ::llvm::ArrayRef<int64_t> strides) {
  return Base::getChecked(emitError, context, offset, strides);
}

::llvm::LogicalResult StridedLayoutAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t offset, ::llvm::ArrayRef<int64_t> strides) {
  if (::mlir::failed(verify(emitError, offset, strides)))
    return ::mlir::failure();
  return ::mlir::success();
}

int64_t StridedLayoutAttr::getOffset() const {
  return getImpl()->offset;
}

::llvm::ArrayRef<int64_t> StridedLayoutAttr::getStrides() const {
  return getImpl()->strides;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::StridedLayoutAttr)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::StringAttr)
namespace mlir {
namespace detail {
struct SymbolRefAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::ArrayRef<FlatSymbolRefAttr>>;
  SymbolRefAttrStorage(StringAttr rootReference, ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences) : rootReference(std::move(rootReference)), nestedReferences(std::move(nestedReferences)) {}

  KeyTy getAsKey() const {
    return KeyTy(rootReference, nestedReferences);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (rootReference == std::get<0>(tblgenKey)) && (nestedReferences == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static SymbolRefAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto rootReference = std::move(std::get<0>(tblgenKey));
    auto nestedReferences = std::move(std::get<1>(tblgenKey));
    nestedReferences = allocator.copyInto(nestedReferences);
    return new (allocator.allocate<SymbolRefAttrStorage>()) SymbolRefAttrStorage(std::move(rootReference), std::move(nestedReferences));
  }

  StringAttr rootReference;
  ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences;
};
} // namespace detail
SymbolRefAttr SymbolRefAttr::get(StringAttr rootReference, ArrayRef<FlatSymbolRefAttr> nestedReferences) {
  return Base::get(rootReference.getContext(), rootReference, nestedReferences);
}

StringAttr SymbolRefAttr::getRootReference() const {
  return getImpl()->rootReference;
}

::llvm::ArrayRef<FlatSymbolRefAttr> SymbolRefAttr::getNestedReferences() const {
  return getImpl()->nestedReferences;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::SymbolRefAttr)
namespace mlir {
namespace detail {
struct TypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Type>;
  TypeAttrStorage(Type value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TypeAttrStorage>()) TypeAttrStorage(std::move(value));
  }

  Type value;
};
} // namespace detail
TypeAttr TypeAttr::get(Type type) {
  return Base::get(type.getContext(), type);
}

Type TypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TypeAttr)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::UnitAttr)

#endif  // GET_ATTRDEF_CLASSES

