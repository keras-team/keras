/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
class ComplexType;
class Float8E5M2Type;
class Float8E4M3Type;
class Float8E4M3FNType;
class Float8E5M2FNUZType;
class Float8E4M3FNUZType;
class Float8E4M3B11FNUZType;
class Float8E3M4Type;
class Float6E2M3FNType;
class Float6E3M2FNType;
class BFloat16Type;
class Float16Type;
class FloatTF32Type;
class Float32Type;
class Float64Type;
class Float80Type;
class Float128Type;
class FunctionType;
class IndexType;
class IntegerType;
class MemRefType;
class NoneType;
class OpaqueType;
class RankedTensorType;
class TupleType;
class UnrankedMemRefType;
class UnrankedTensorType;
class VectorType;
namespace detail {
struct ComplexTypeStorage;
} // namespace detail
class ComplexType : public ::mlir::Type::TypeBase<ComplexType, ::mlir::Type, detail::ComplexTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "builtin.complex";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static ComplexType get(Type elementType);
  static ComplexType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  Type getElementType() const;
};
class Float8E5M2Type : public ::mlir::Type::TypeBase<Float8E5M2Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E5M2Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E5M2";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float8E4M3Type : public ::mlir::Type::TypeBase<Float8E4M3Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E4M3Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E4M3";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float8E4M3FNType : public ::mlir::Type::TypeBase<Float8E4M3FNType, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E4M3FNType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E4M3FN";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float8E5M2FNUZType : public ::mlir::Type::TypeBase<Float8E5M2FNUZType, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E5M2FNUZType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E5M2FNUZ";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float8E4M3FNUZType : public ::mlir::Type::TypeBase<Float8E4M3FNUZType, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E4M3FNUZType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E4M3FNUZ";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float8E4M3B11FNUZType : public ::mlir::Type::TypeBase<Float8E4M3B11FNUZType, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E4M3B11FNUZType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E4M3B11FNUZ";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float8E3M4Type : public ::mlir::Type::TypeBase<Float8E3M4Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float8E3M4Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f8E3M4";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float6E2M3FNType : public ::mlir::Type::TypeBase<Float6E2M3FNType, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float6E2M3FNType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f6E2M3FN";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float6E3M2FNType : public ::mlir::Type::TypeBase<Float6E3M2FNType, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float6E3M2FNType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f6E3M2FN";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class BFloat16Type : public ::mlir::Type::TypeBase<BFloat16Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static BFloat16Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.bf16";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float16Type : public ::mlir::Type::TypeBase<Float16Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float16Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f16";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class FloatTF32Type : public ::mlir::Type::TypeBase<FloatTF32Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static FloatTF32Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.tf32";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float32Type : public ::mlir::Type::TypeBase<Float32Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float32Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f32";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float64Type : public ::mlir::Type::TypeBase<Float64Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float64Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f64";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float80Type : public ::mlir::Type::TypeBase<Float80Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float80Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f80";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class Float128Type : public ::mlir::Type::TypeBase<Float128Type, ::mlir::FloatType, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static Float128Type get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.f128";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class FunctionType : public ::mlir::Type::TypeBase<FunctionType, ::mlir::Type, detail::FunctionTypeStorage> {
public:
  using Base::Base;
  /// Input types.
  unsigned getNumInputs() const;
  Type getInput(unsigned i) const { return getInputs()[i]; }

  /// Result types.
  unsigned getNumResults() const;
  Type getResult(unsigned i) const { return getResults()[i]; }

  /// Returns a clone of this function type with the given argument
  /// and result types.
  FunctionType clone(TypeRange inputs, TypeRange results) const;

  /// Returns a new function type with the specified arguments and results
  /// inserted.
  FunctionType getWithArgsAndResults(ArrayRef<unsigned> argIndices,
                                     TypeRange argTypes,
                                     ArrayRef<unsigned> resultIndices,
                                     TypeRange resultTypes);

  /// Returns a new function type without the specified arguments and results.
  FunctionType getWithoutArgsAndResults(const BitVector &argIndices,
                                        const BitVector &resultIndices);
  static constexpr ::llvm::StringLiteral name = "builtin.function";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static FunctionType get(::mlir::MLIRContext *context, TypeRange inputs, TypeRange results);
  ArrayRef<Type> getInputs() const;
  ArrayRef<Type> getResults() const;
};
class IndexType : public ::mlir::Type::TypeBase<IndexType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static IndexType get(MLIRContext *context);

  /// Storage bit width used for IndexType by internal compiler data
  /// structures.
  static constexpr unsigned kInternalStorageBitWidth = 64;
  static constexpr ::llvm::StringLiteral name = "builtin.index";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
class IntegerType : public ::mlir::Type::TypeBase<IntegerType, ::mlir::Type, detail::IntegerTypeStorage> {
public:
  using Base::Base;
  /// Signedness semantics.
  enum SignednessSemantics : uint32_t {
    Signless, /// No signedness semantics
    Signed,   /// Signed integer
    Unsigned, /// Unsigned integer
  };

  /// Return true if this is a signless integer type.
  bool isSignless() const { return getSignedness() == Signless; }
  /// Return true if this is a signed integer type.
  bool isSigned() const { return getSignedness() == Signed; }
  /// Return true if this is an unsigned integer type.
  bool isUnsigned() const { return getSignedness() == Unsigned; }

  /// Get or create a new IntegerType with the same signedness as `this` and a
  /// bitwidth scaled by `scale`.
  /// Return null if the scaled element type cannot be represented.
  IntegerType scaleElementBitwidth(unsigned scale);

  /// Integer representation maximal bitwidth.
  /// Note: This is aligned with the maximum width of llvm::IntegerType.
  static constexpr unsigned kMaxWidth = (1 << 24) - 1;
  static constexpr ::llvm::StringLiteral name = "builtin.integer";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static IntegerType get(::mlir::MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);
  static IntegerType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, SignednessSemantics signedness);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, SignednessSemantics signedness);
  unsigned getWidth() const;
  SignednessSemantics getSignedness() const;
};
namespace detail {
struct MemRefTypeStorage;
} // namespace detail
class MemRefType : public ::mlir::Type::TypeBase<MemRefType, BaseMemRefType, detail::MemRefTypeStorage, ::mlir::ShapedType::Trait> {
public:
  using Base::Base;
  using BaseMemRefType::clone;
  using ShapedType::Trait<MemRefType>::getElementTypeBitWidth;
  using ShapedType::Trait<MemRefType>::getRank;
  using ShapedType::Trait<MemRefType>::getNumElements;
  using ShapedType::Trait<MemRefType>::isDynamicDim;
  using ShapedType::Trait<MemRefType>::hasStaticShape;
  using ShapedType::Trait<MemRefType>::getNumDynamicDims;
  using ShapedType::Trait<MemRefType>::getDimSize;
  using ShapedType::Trait<MemRefType>::getDynamicDimIndex;

  /// This is a builder type that keeps local references to arguments.
  /// Arguments that are passed into the builder must outlive the builder.
  class Builder;

  /// [deprecated] Returns the memory space in old raw integer representation.
  /// New `Attribute getMemorySpace()` method should be used instead.
  unsigned getMemorySpaceAsInt() const;

  static constexpr ::llvm::StringLiteral name = "builtin.memref";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static MemRefType get(ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout = {}, Attribute memorySpace = {});
  static MemRefType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout = {}, Attribute memorySpace = {});
  static MemRefType get(ArrayRef<int64_t> shape, Type elementType, AffineMap map, Attribute memorySpace = {});
  static MemRefType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, AffineMap map, Attribute memorySpace = {});
  static MemRefType get(ArrayRef<int64_t> shape, Type elementType, AffineMap map, unsigned memorySpaceInd);
  static MemRefType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, AffineMap map, unsigned memorySpaceInd);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace);
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  MemRefLayoutAttrInterface getLayout() const;
  Attribute getMemorySpace() const;
};
class NoneType : public ::mlir::Type::TypeBase<NoneType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static NoneType get(MLIRContext *context);
  static constexpr ::llvm::StringLiteral name = "builtin.none";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
};
namespace detail {
struct OpaqueTypeStorage;
} // namespace detail
class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type, detail::OpaqueTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "builtin.opaque";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static OpaqueType get(StringAttr dialectNamespace, StringRef typeData = {});
  static OpaqueType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, StringRef typeData = {});
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, ::llvm::StringRef typeData);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, ::llvm::StringRef typeData);
  StringAttr getDialectNamespace() const;
  ::llvm::StringRef getTypeData() const;
};
namespace detail {
struct RankedTensorTypeStorage;
} // namespace detail
class RankedTensorType : public ::mlir::Type::TypeBase<RankedTensorType, TensorType, detail::RankedTensorTypeStorage, ::mlir::ShapedType::Trait, ::mlir::ValueSemantics> {
public:
  using Base::Base;
  using TensorType::clone;
  using ShapedType::Trait<RankedTensorType>::getElementTypeBitWidth;
  using ShapedType::Trait<RankedTensorType>::getRank;
  using ShapedType::Trait<RankedTensorType>::getNumElements;
  using ShapedType::Trait<RankedTensorType>::isDynamicDim;
  using ShapedType::Trait<RankedTensorType>::hasStaticShape;
  using ShapedType::Trait<RankedTensorType>::getNumDynamicDims;
  using ShapedType::Trait<RankedTensorType>::getDimSize;
  using ShapedType::Trait<RankedTensorType>::getDynamicDimIndex;

  /// This is a builder type that keeps local references to arguments.
  /// Arguments that are passed into the builder must outlive the builder.
  class Builder;

  /// Return a clone of this type with the given new element type and the same
  /// shape as this type.
  RankedTensorType clone(::mlir::Type elementType) {
    return ::llvm::cast<RankedTensorType>(cloneWith(getShape(), elementType));
  }
  static constexpr ::llvm::StringLiteral name = "builtin.tensor";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static RankedTensorType get(ArrayRef<int64_t> shape, Type elementType, Attribute encoding = {});
  static RankedTensorType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, Attribute encoding = {});
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding);
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  Attribute getEncoding() const;
};
class TupleType : public ::mlir::Type::TypeBase<TupleType, ::mlir::Type, detail::TupleTypeStorage> {
public:
  using Base::Base;
  /// Accumulate the types contained in this tuple and tuples nested within
  /// it. Note that this only flattens nested tuples, not any other container
  /// type, e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is
  /// flattened to (i32, tensor<i32>, f32, i64)
  void getFlattenedTypes(SmallVectorImpl<Type> &types);

  /// Return the number of held types.
  size_t size() const;

  /// Iterate over the held elements.
  using iterator = ArrayRef<Type>::iterator;
  iterator begin() const { return getTypes().begin(); }
  iterator end() const { return getTypes().end(); }

  /// Return the element type at index 'index'.
  Type getType(size_t index) const {
    assert(index < size() && "invalid index for tuple type");
    return getTypes()[index];
  }
  static constexpr ::llvm::StringLiteral name = "builtin.tuple";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  static TupleType get(::mlir::MLIRContext *context, TypeRange elementTypes);
  static TupleType get(::mlir::MLIRContext *context);
  ArrayRef<Type> getTypes() const;
};
namespace detail {
struct UnrankedMemRefTypeStorage;
} // namespace detail
class UnrankedMemRefType : public ::mlir::Type::TypeBase<UnrankedMemRefType, BaseMemRefType, detail::UnrankedMemRefTypeStorage, ::mlir::ShapedType::Trait> {
public:
  using Base::Base;
  using BaseMemRefType::clone;
  using ShapedType::Trait<UnrankedMemRefType>::getElementTypeBitWidth;
  using ShapedType::Trait<UnrankedMemRefType>::getRank;
  using ShapedType::Trait<UnrankedMemRefType>::getNumElements;
  using ShapedType::Trait<UnrankedMemRefType>::isDynamicDim;
  using ShapedType::Trait<UnrankedMemRefType>::hasStaticShape;
  using ShapedType::Trait<UnrankedMemRefType>::getNumDynamicDims;
  using ShapedType::Trait<UnrankedMemRefType>::getDimSize;
  using ShapedType::Trait<UnrankedMemRefType>::getDynamicDimIndex;

  ArrayRef<int64_t> getShape() const { return std::nullopt; }

  /// [deprecated] Returns the memory space in old raw integer representation.
  /// New `Attribute getMemorySpace()` method should be used instead.
  unsigned getMemorySpaceAsInt() const;

  /// Return a clone of this type with the given new element type and the same
  /// shape as this type.
  MemRefType clone(::mlir::Type elementType) {
    return ::llvm::cast<MemRefType>(cloneWith(getShape(), elementType));
  }
  static constexpr ::llvm::StringLiteral name = "builtin.unranked_memref";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static UnrankedMemRefType get(Type elementType, Attribute memorySpace);
  static UnrankedMemRefType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace);
  static UnrankedMemRefType get(Type elementType, unsigned memorySpace);
  static UnrankedMemRefType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned memorySpace);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace);
  Type getElementType() const;
  Attribute getMemorySpace() const;
};
namespace detail {
struct UnrankedTensorTypeStorage;
} // namespace detail
class UnrankedTensorType : public ::mlir::Type::TypeBase<UnrankedTensorType, TensorType, detail::UnrankedTensorTypeStorage, ::mlir::ShapedType::Trait, ::mlir::ValueSemantics> {
public:
  using Base::Base;
  using TensorType::clone;
  using ShapedType::Trait<UnrankedTensorType>::getElementTypeBitWidth;
  using ShapedType::Trait<UnrankedTensorType>::getRank;
  using ShapedType::Trait<UnrankedTensorType>::getNumElements;
  using ShapedType::Trait<UnrankedTensorType>::isDynamicDim;
  using ShapedType::Trait<UnrankedTensorType>::hasStaticShape;
  using ShapedType::Trait<UnrankedTensorType>::getNumDynamicDims;
  using ShapedType::Trait<UnrankedTensorType>::getDimSize;
  using ShapedType::Trait<UnrankedTensorType>::getDynamicDimIndex;

  ArrayRef<int64_t> getShape() const { return std::nullopt; }
  static constexpr ::llvm::StringLiteral name = "builtin.unranked_tensor";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static UnrankedTensorType get(Type elementType);
  static UnrankedTensorType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  Type getElementType() const;
};
namespace detail {
struct VectorTypeStorage;
} // namespace detail
class VectorType : public ::mlir::Type::TypeBase<VectorType, Type, detail::VectorTypeStorage, ::mlir::ShapedType::Trait, ::mlir::ValueSemantics> {
public:
  using Base::Base;
  /// This is a builder type that keeps local references to arguments.
  /// Arguments that are passed into the builder must outlive the builder.
  class Builder;

  /// Returns true if the given type can be used as an element of a vector
  /// type. See "Builtin_VectorTypeElementType" for allowed types.
  static bool isValidElementType(Type t);

  /// Returns true if the vector contains scalable dimensions.
  bool isScalable() const {
    return llvm::is_contained(getScalableDims(), true);
  }
  bool allDimsScalable() const {
    // Treat 0-d vectors as fixed size.
    if (getRank() == 0)
      return false;
    return !llvm::is_contained(getScalableDims(), false);
  }

  /// Get the number of scalable dimensions.
  int64_t getNumScalableDims() const {
    return llvm::count(getScalableDims(), true);
  }

  /// Get or create a new VectorType with the same shape as `this` and an
  /// element type of bitwidth scaled by `scale`.
  /// Return null if the scaled element type cannot be represented.
  VectorType scaleElementBitwidth(unsigned scale);

  /// Returns if this type is ranked (always true).
  bool hasRank() const { return true; }

  /// Clone this vector type with the given shape and element type. If the
  /// provided shape is `std::nullopt`, the current shape of the type is used.
  VectorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const;
  static constexpr ::llvm::StringLiteral name = "builtin.vector";
  static constexpr ::llvm::StringLiteral dialectName = "builtin";
  using Base::getChecked;
  static VectorType get(ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims = {});
  static VectorType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims = {});
  static ::llvm::LogicalResult verifyInvariantsImpl(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::ArrayRef<bool> scalableDims);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::ArrayRef<bool> scalableDims);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::ArrayRef<bool> scalableDims);
  ::llvm::ArrayRef<int64_t> getShape() const;
  ::mlir::Type getElementType() const;
  ::llvm::ArrayRef<bool> getScalableDims() const;
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ComplexType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E5M2Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3FNType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E5M2FNUZType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3FNUZType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E4M3B11FNUZType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float8E3M4Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float6E2M3FNType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float6E3M2FNType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::BFloat16Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float16Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::FloatTF32Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float32Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float64Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float80Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::Float128Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::FunctionType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::IndexType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::MemRefType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NoneType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::OpaqueType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::RankedTensorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TupleType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::UnrankedMemRefType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::UnrankedTensorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::VectorType)

#endif  // GET_TYPEDEF_CLASSES

