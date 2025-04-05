/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::ComplexType,
::mlir::Float8E5M2Type,
::mlir::Float8E4M3Type,
::mlir::Float8E4M3FNType,
::mlir::Float8E5M2FNUZType,
::mlir::Float8E4M3FNUZType,
::mlir::Float8E4M3B11FNUZType,
::mlir::Float8E3M4Type,
::mlir::Float6E2M3FNType,
::mlir::Float6E3M2FNType,
::mlir::BFloat16Type,
::mlir::Float16Type,
::mlir::FloatTF32Type,
::mlir::Float32Type,
::mlir::Float64Type,
::mlir::Float80Type,
::mlir::Float128Type,
::mlir::FunctionType,
::mlir::IndexType,
::mlir::IntegerType,
::mlir::MemRefType,
::mlir::NoneType,
::mlir::OpaqueType,
::mlir::RankedTensorType,
::mlir::TupleType,
::mlir::UnrankedMemRefType,
::mlir::UnrankedTensorType,
::mlir::VectorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

namespace mlir {
namespace detail {
struct ComplexTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  ComplexTypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComplexTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComplexTypeStorage>()) ComplexTypeStorage(std::move(elementType));
  }

  Type elementType;
};
} // namespace detail
ComplexType ComplexType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

ComplexType ComplexType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

::llvm::LogicalResult ComplexType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  if (::mlir::failed(verify(emitError, elementType)))
    return ::mlir::failure();
  return ::mlir::success();
}

Type ComplexType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ComplexType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E5M2Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3FNType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E5M2FNUZType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3FNUZType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3B11FNUZType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float8E3M4Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float6E2M3FNType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float6E3M2FNType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::BFloat16Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float16Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::FloatTF32Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float32Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float64Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float80Type)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::Float128Type)
namespace mlir {
FunctionType FunctionType::get(::mlir::MLIRContext *context, TypeRange inputs, TypeRange results) {
  return Base::get(context, inputs, results);
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::FunctionType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IndexType)
namespace mlir {
::llvm::LogicalResult IntegerType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, SignednessSemantics signedness) {
  if (::mlir::failed(verify(emitError, width, signedness)))
    return ::mlir::failure();
  return ::mlir::success();
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
namespace mlir {
namespace detail {
struct MemRefTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, MemRefLayoutAttrInterface, Attribute>;
  MemRefTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace) : shape(std::move(shape)), elementType(std::move(elementType)), layout(std::move(layout)), memorySpace(std::move(memorySpace)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, layout, memorySpace);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (layout == std::get<2>(tblgenKey)) && (memorySpace == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static MemRefTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto layout = std::move(std::get<2>(tblgenKey));
    auto memorySpace = std::move(std::get<3>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<MemRefTypeStorage>()) MemRefTypeStorage(std::move(shape), std::move(elementType), std::move(layout), std::move(memorySpace));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  Attribute memorySpace;
};
} // namespace detail
::llvm::LogicalResult MemRefType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace) {
  if (::mlir::failed(verify(emitError, shape, elementType, layout, memorySpace)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<int64_t> MemRefType::getShape() const {
  return getImpl()->shape;
}

Type MemRefType::getElementType() const {
  return getImpl()->elementType;
}

MemRefLayoutAttrInterface MemRefType::getLayout() const {
  return getImpl()->layout;
}

Attribute MemRefType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::MemRefType)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NoneType)
namespace mlir {
namespace detail {
struct OpaqueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::StringRef>;
  OpaqueTypeStorage(StringAttr dialectNamespace, ::llvm::StringRef typeData) : dialectNamespace(std::move(dialectNamespace)), typeData(std::move(typeData)) {}

  KeyTy getAsKey() const {
    return KeyTy(dialectNamespace, typeData);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (dialectNamespace == std::get<0>(tblgenKey)) && (typeData == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static OpaqueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto dialectNamespace = std::move(std::get<0>(tblgenKey));
    auto typeData = std::move(std::get<1>(tblgenKey));
    typeData = allocator.copyInto(typeData);
    return new (allocator.allocate<OpaqueTypeStorage>()) OpaqueTypeStorage(std::move(dialectNamespace), std::move(typeData));
  }

  StringAttr dialectNamespace;
  ::llvm::StringRef typeData;
};
} // namespace detail
OpaqueType OpaqueType::get(StringAttr dialectNamespace, StringRef typeData) {
  return Base::get(dialectNamespace.getContext(), dialectNamespace, typeData);
}

OpaqueType OpaqueType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, StringRef typeData) {
  return Base::getChecked(emitError, dialectNamespace.getContext(), dialectNamespace, typeData);
}

::llvm::LogicalResult OpaqueType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, ::llvm::StringRef typeData) {
  if (::mlir::failed(verify(emitError, dialectNamespace, typeData)))
    return ::mlir::failure();
  return ::mlir::success();
}

StringAttr OpaqueType::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

::llvm::StringRef OpaqueType::getTypeData() const {
  return getImpl()->typeData;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::OpaqueType)
namespace mlir {
namespace detail {
struct RankedTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, Attribute>;
  RankedTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding) : shape(std::move(shape)), elementType(std::move(elementType)), encoding(std::move(encoding)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, encoding);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (encoding == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static RankedTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto encoding = std::move(std::get<2>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<RankedTensorTypeStorage>()) RankedTensorTypeStorage(std::move(shape), std::move(elementType), std::move(encoding));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  Attribute encoding;
};
} // namespace detail
RankedTensorType RankedTensorType::get(ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  return Base::get(elementType.getContext(), shape, elementType, encoding);
}

RankedTensorType RankedTensorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  return Base::getChecked(emitError, elementType.getContext(), shape, elementType, encoding);
}

::llvm::LogicalResult RankedTensorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  if (::mlir::failed(verify(emitError, shape, elementType, encoding)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<int64_t> RankedTensorType::getShape() const {
  return getImpl()->shape;
}

Type RankedTensorType::getElementType() const {
  return getImpl()->elementType;
}

Attribute RankedTensorType::getEncoding() const {
  return getImpl()->encoding;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::RankedTensorType)
namespace mlir {
TupleType TupleType::get(::mlir::MLIRContext *context, TypeRange elementTypes) {
  return Base::get(context, elementTypes);
}

TupleType TupleType::get(::mlir::MLIRContext *context) {
  return Base::get(context, TypeRange());
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TupleType)
namespace mlir {
namespace detail {
struct UnrankedMemRefTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, Attribute>;
  UnrankedMemRefTypeStorage(Type elementType, Attribute memorySpace) : elementType(std::move(elementType)), memorySpace(std::move(memorySpace)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, memorySpace);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (memorySpace == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static UnrankedMemRefTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    auto memorySpace = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<UnrankedMemRefTypeStorage>()) UnrankedMemRefTypeStorage(std::move(elementType), std::move(memorySpace));
  }

  Type elementType;
  Attribute memorySpace;
};
} // namespace detail
UnrankedMemRefType UnrankedMemRefType::get(Type elementType, Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
  return Base::get(elementType.getContext(), elementType, nonDefaultMemorySpace);
}

UnrankedMemRefType UnrankedMemRefType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
  return Base::getChecked(emitError, elementType.getContext(), elementType, nonDefaultMemorySpace);
}

UnrankedMemRefType UnrankedMemRefType::get(Type elementType, unsigned memorySpace) {
  // Convert deprecated integer-like memory space to Attribute.
  Attribute memorySpaceAttr =
      wrapIntegerMemorySpace(memorySpace, elementType.getContext());
  return UnrankedMemRefType::get(elementType, memorySpaceAttr);
}

UnrankedMemRefType UnrankedMemRefType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned memorySpace) {
  // Convert deprecated integer-like memory space to Attribute.
  Attribute memorySpaceAttr =
      wrapIntegerMemorySpace(memorySpace, elementType.getContext());
  return UnrankedMemRefType::get(elementType, memorySpaceAttr);
}

::llvm::LogicalResult UnrankedMemRefType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace) {
  if (::mlir::failed(verify(emitError, elementType, memorySpace)))
    return ::mlir::failure();
  return ::mlir::success();
}

Type UnrankedMemRefType::getElementType() const {
  return getImpl()->elementType;
}

Attribute UnrankedMemRefType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::UnrankedMemRefType)
namespace mlir {
namespace detail {
struct UnrankedTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  UnrankedTensorTypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static UnrankedTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<UnrankedTensorTypeStorage>()) UnrankedTensorTypeStorage(std::move(elementType));
  }

  Type elementType;
};
} // namespace detail
UnrankedTensorType UnrankedTensorType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

UnrankedTensorType UnrankedTensorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

::llvm::LogicalResult UnrankedTensorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  if (::mlir::failed(verify(emitError, elementType)))
    return ::mlir::failure();
  return ::mlir::success();
}

Type UnrankedTensorType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::UnrankedTensorType)
namespace mlir {
namespace detail {
struct VectorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::mlir::Type, ::llvm::ArrayRef<bool>>;
  VectorTypeStorage(::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::ArrayRef<bool> scalableDims) : shape(std::move(shape)), elementType(std::move(elementType)), scalableDims(std::move(scalableDims)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, scalableDims);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (scalableDims == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static VectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto scalableDims = std::move(std::get<2>(tblgenKey));
    shape = allocator.copyInto(shape);
    scalableDims = allocator.copyInto(scalableDims);
    return new (allocator.allocate<VectorTypeStorage>()) VectorTypeStorage(std::move(shape), std::move(elementType), std::move(scalableDims));
  }

  ::llvm::ArrayRef<int64_t> shape;
  ::mlir::Type elementType;
  ::llvm::ArrayRef<bool> scalableDims;
};
} // namespace detail
VectorType VectorType::get(ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims) {
  // While `scalableDims` is optional, its default value should be
  // `false` for every dim in `shape`.
  SmallVector<bool> isScalableVec;
  if (scalableDims.empty()) {
    isScalableVec.resize(shape.size(), false);
    scalableDims = isScalableVec;
  }
  return Base::get(elementType.getContext(), shape, elementType, scalableDims);
}

VectorType VectorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims) {
  // While `scalableDims` is optional, its default value should be
  // `false` for every dim in `shape`.
  SmallVector<bool> isScalableVec;
  if (scalableDims.empty()) {
    isScalableVec.resize(shape.size(), false);
    scalableDims = isScalableVec;
  }
  return Base::getChecked(emitError, elementType.getContext(), shape, elementType, scalableDims);
}

::llvm::LogicalResult VectorType::verifyInvariantsImpl(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::ArrayRef<bool> scalableDims) {
  if (!(((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))))) {
    emitError() << "failed to verify 'elementType': integer or index or floating-point";
    return ::mlir::failure();
  }

  return ::mlir::success();
}

::llvm::LogicalResult VectorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::ArrayRef<bool> scalableDims) {
  if (::mlir::failed(verifyInvariantsImpl(emitError, shape, elementType, scalableDims)))
    return ::mlir::failure();
  if (::mlir::failed(verify(emitError, shape, elementType, scalableDims)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<int64_t> VectorType::getShape() const {
  return getImpl()->shape;
}

::mlir::Type VectorType::getElementType() const {
  return getImpl()->elementType;
}

::llvm::ArrayRef<bool> VectorType::getScalableDims() const {
  return getImpl()->scalableDims;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::VectorType)

#endif  // GET_TYPEDEF_CLASSES

