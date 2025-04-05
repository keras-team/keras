//===- BuiltinTypes.h - MLIR Builtin Type Classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPES_H
#define MLIR_IR_BUILTINTYPES_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/ADTExtras.h"

namespace llvm {
class BitVector;
struct fltSemantics;
} // namespace llvm

//===----------------------------------------------------------------------===//
// Tablegen Interface Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class AffineExpr;
class AffineMap;
class FloatType;
class IndexType;
class IntegerType;
class MemRefType;
class RankedTensorType;
class StringAttr;
class TypeRange;

namespace detail {
struct FunctionTypeStorage;
struct IntegerTypeStorage;
struct TupleTypeStorage;
} // namespace detail

/// Type trait indicating that the type has value semantics.
template <typename ConcreteType>
class ValueSemantics
    : public TypeTrait::TraitBase<ConcreteType, ValueSemantics> {};

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

class FloatType : public Type {
public:
  using Type::Type;

  // Convenience factories.
  static FloatType getBF16(MLIRContext *ctx);
  static FloatType getF16(MLIRContext *ctx);
  static FloatType getF32(MLIRContext *ctx);
  static FloatType getTF32(MLIRContext *ctx);
  static FloatType getF64(MLIRContext *ctx);
  static FloatType getF80(MLIRContext *ctx);
  static FloatType getF128(MLIRContext *ctx);
  static FloatType getFloat8E5M2(MLIRContext *ctx);
  static FloatType getFloat8E4M3(MLIRContext *ctx);
  static FloatType getFloat8E4M3FN(MLIRContext *ctx);
  static FloatType getFloat8E5M2FNUZ(MLIRContext *ctx);
  static FloatType getFloat8E4M3FNUZ(MLIRContext *ctx);
  static FloatType getFloat8E4M3B11FNUZ(MLIRContext *ctx);
  static FloatType getFloat8E3M4(MLIRContext *ctx);
  static FloatType getFloat6E2M3FN(MLIRContext *ctx);
  static FloatType getFloat6E3M2FN(MLIRContext *ctx);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Return the bitwidth of this float type.
  unsigned getWidth();

  /// Return the width of the mantissa of this type.
  /// The width includes the integer bit.
  unsigned getFPMantissaWidth();

  /// Get or create a new FloatType with bitwidth scaled by `scale`.
  /// Return null if the scaled element type cannot be represented.
  FloatType scaleElementBitwidth(unsigned scale);

  /// Return the floating semantics of this float type.
  const llvm::fltSemantics &getFloatSemantics();
};

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// Tensor types represent multi-dimensional arrays, and have two variants:
/// RankedTensorType and UnrankedTensorType.
/// Note: This class attaches the ShapedType trait to act as a mixin to
///       provide many useful utility functions. This inheritance has no effect
///       on derived tensor types.
class TensorType : public Type, public ShapedType::Trait<TensorType> {
public:
  using Type::Type;

  /// Returns the element type of this tensor type.
  Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this tensor type.
  ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `std::nullopt`, the current shape of the type is used.
  TensorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const;

  // Make sure that base class overloads are visible.
  using ShapedType::Trait<TensorType>::clone;

  /// Return a clone of this type with the given new shape and element type.
  /// The returned type is ranked, even if this type is unranked.
  RankedTensorType clone(ArrayRef<int64_t> shape, Type elementType) const;

  /// Return a clone of this type with the given new shape. The returned type
  /// is ranked, even if this type is unranked.
  RankedTensorType clone(ArrayRef<int64_t> shape) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator ShapedType() const { return llvm::cast<ShapedType>(*this); }
};

//===----------------------------------------------------------------------===//
// BaseMemRefType
//===----------------------------------------------------------------------===//

/// This class provides a shared interface for ranked and unranked memref types.
/// Note: This class attaches the ShapedType trait to act as a mixin to
///       provide many useful utility functions. This inheritance has no effect
///       on derived memref types.
class BaseMemRefType : public Type, public ShapedType::Trait<BaseMemRefType> {
public:
  using Type::Type;

  /// Returns the element type of this memref type.
  Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this memref type.
  ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `std::nullopt`, the current shape of the type is used.
  BaseMemRefType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                           Type elementType) const;

  // Make sure that base class overloads are visible.
  using ShapedType::Trait<BaseMemRefType>::clone;

  /// Return a clone of this type with the given new shape and element type.
  /// The returned type is ranked, even if this type is unranked.
  MemRefType clone(ArrayRef<int64_t> shape, Type elementType) const;

  /// Return a clone of this type with the given new shape. The returned type
  /// is ranked, even if this type is unranked.
  MemRefType clone(ArrayRef<int64_t> shape) const;

  /// Return true if the specified element type is ok in a memref.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Returns the memory space in which data referred to by this memref resides.
  Attribute getMemorySpace() const;

  /// [deprecated] Returns the memory space in old raw integer representation.
  /// New `Attribute getMemorySpace()` method should be used instead.
  unsigned getMemorySpaceAsInt() const;

  /// Allow implicit conversion to ShapedType.
  operator ShapedType() const { return llvm::cast<ShapedType>(*this); }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/IR/BuiltinTypes.h.inc"

namespace mlir {
#include "mlir/IR/BuiltinTypeConstraints.h.inc"

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class MemRefType::Builder {
public:
  // Build from another MemRefType.
  explicit Builder(MemRefType other)
      : shape(other.getShape()), elementType(other.getElementType()),
        layout(other.getLayout()), memorySpace(other.getMemorySpace()) {}

  // Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType)
      : shape(shape), elementType(elementType) {}

  Builder &setShape(ArrayRef<int64_t> newShape) {
    shape = newShape;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  Builder &setLayout(MemRefLayoutAttrInterface newLayout) {
    layout = newLayout;
    return *this;
  }

  Builder &setMemorySpace(Attribute newMemorySpace) {
    memorySpace = newMemorySpace;
    return *this;
  }

  operator MemRefType() {
    return MemRefType::get(shape, elementType, layout, memorySpace);
  }

private:
  ArrayRef<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  Attribute memorySpace;
};

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class RankedTensorType::Builder {
public:
  /// Build from another RankedTensorType.
  explicit Builder(RankedTensorType other)
      : shape(other.getShape()), elementType(other.getElementType()),
        encoding(other.getEncoding()) {}

  /// Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType, Attribute encoding)
      : shape(shape), elementType(elementType), encoding(encoding) {}

  Builder &setShape(ArrayRef<int64_t> newShape) {
    shape = newShape;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  Builder &setEncoding(Attribute newEncoding) {
    encoding = newEncoding;
    return *this;
  }

  /// Erase a dim from shape @pos.
  Builder &dropDim(unsigned pos) {
    assert(pos < shape.size() && "overflow");
    shape.erase(pos);
    return *this;
  }

  /// Insert a val into shape @pos.
  Builder &insertDim(int64_t val, unsigned pos) {
    assert(pos <= shape.size() && "overflow");
    shape.insert(pos, val);
    return *this;
  }

  operator RankedTensorType() {
    return RankedTensorType::get(shape, elementType, encoding);
  }

private:
  CopyOnWriteArrayRef<int64_t> shape;
  Type elementType;
  Attribute encoding;
};

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class VectorType::Builder {
public:
  /// Build from another VectorType.
  explicit Builder(VectorType other)
      : elementType(other.getElementType()), shape(other.getShape()),
        scalableDims(other.getScalableDims()) {}

  /// Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType,
          ArrayRef<bool> scalableDims = {})
      : elementType(elementType), shape(shape), scalableDims(scalableDims) {}

  Builder &setShape(ArrayRef<int64_t> newShape,
                    ArrayRef<bool> newIsScalableDim = {}) {
    shape = newShape;
    scalableDims = newIsScalableDim;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  /// Erase a dim from shape @pos.
  Builder &dropDim(unsigned pos) {
    assert(pos < shape.size() && "overflow");
    shape.erase(pos);
    if (!scalableDims.empty())
      scalableDims.erase(pos);
    return *this;
  }

  /// Set a dim in shape @pos to val.
  Builder &setDim(unsigned pos, int64_t val) {
    assert(pos < shape.size() && "overflow");
    shape.set(pos, val);
    return *this;
  }

  operator VectorType() {
    return VectorType::get(shape, elementType, scalableDims);
  }

private:
  Type elementType;
  CopyOnWriteArrayRef<int64_t> shape;
  CopyOnWriteArrayRef<bool> scalableDims;
};

/// Given an `originalShape` and a `reducedShape` assumed to be a subset of
/// `originalShape` with some `1` entries erased, return the set of indices
/// that specifies which of the entries of `originalShape` are dropped to obtain
/// `reducedShape`. The returned mask can be applied as a projection to
/// `originalShape` to obtain the `reducedShape`. This mask is useful to track
/// which dimensions must be kept when e.g. compute MemRef strides under
/// rank-reducing operations. Return std::nullopt if reducedShape cannot be
/// obtained by dropping only `1` entries in `originalShape`.
/// If `matchDynamic` is true, then dynamic dims in `originalShape` and
/// `reducedShape` will be considered matching with non-dynamic dims, unless
/// the non-dynamic dim is from `originalShape` and equal to 1. For example,
/// in ([1, 3, ?], [?, 5]), the mask would be {1, 0, 0}, since 3 and 5 will
/// match with the corresponding dynamic dims.
std::optional<llvm::SmallDenseSet<unsigned>>
computeRankReductionMask(ArrayRef<int64_t> originalShape,
                         ArrayRef<int64_t> reducedShape,
                         bool matchDynamic = false);

/// Enum that captures information related to verifier error conditions on
/// slice insert/extract type of ops.
enum class SliceVerificationResult {
  Success,
  RankTooLarge,
  SizeMismatch,
  ElemTypeMismatch,
  // Error codes to ops with a memory space and a layout annotation.
  MemSpaceMismatch,
  LayoutMismatch
};

/// Check if `originalType` can be rank reduced to `candidateReducedType` type
/// by dropping some dimensions with static size `1`.
/// Return `SliceVerificationResult::Success` on success or an appropriate error
/// code.
SliceVerificationResult isRankReducedType(ShapedType originalType,
                                          ShapedType candidateReducedType);

//===----------------------------------------------------------------------===//
// Deferred Method Definitions
//===----------------------------------------------------------------------===//

inline bool BaseMemRefType::classof(Type type) {
  return llvm::isa<MemRefType, UnrankedMemRefType>(type);
}

inline bool BaseMemRefType::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat() ||
         llvm::isa<ComplexType, MemRefType, VectorType, UnrankedMemRefType>(
             type) ||
         llvm::isa<MemRefElementTypeInterface>(type);
}

inline bool FloatType::classof(Type type) {
  return llvm::isa<Float6E2M3FNType, Float6E3M2FNType, Float8E5M2Type,
                   Float8E4M3Type, Float8E4M3FNType, Float8E5M2FNUZType,
                   Float8E4M3FNUZType, Float8E4M3B11FNUZType, Float8E3M4Type,
                   BFloat16Type, Float16Type, FloatTF32Type, Float32Type,
                   Float64Type, Float80Type, Float128Type>(type);
}

inline FloatType FloatType::getFloat6E2M3FN(MLIRContext *ctx) {
  return Float6E2M3FNType::get(ctx);
}

inline FloatType FloatType::getFloat6E3M2FN(MLIRContext *ctx) {
  return Float6E3M2FNType::get(ctx);
}

inline FloatType FloatType::getFloat8E5M2(MLIRContext *ctx) {
  return Float8E5M2Type::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3(MLIRContext *ctx) {
  return Float8E4M3Type::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3FN(MLIRContext *ctx) {
  return Float8E4M3FNType::get(ctx);
}

inline FloatType FloatType::getFloat8E5M2FNUZ(MLIRContext *ctx) {
  return Float8E5M2FNUZType::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3FNUZ(MLIRContext *ctx) {
  return Float8E4M3FNUZType::get(ctx);
}

inline FloatType FloatType::getFloat8E4M3B11FNUZ(MLIRContext *ctx) {
  return Float8E4M3B11FNUZType::get(ctx);
}

inline FloatType FloatType::getFloat8E3M4(MLIRContext *ctx) {
  return Float8E3M4Type::get(ctx);
}

inline FloatType FloatType::getBF16(MLIRContext *ctx) {
  return BFloat16Type::get(ctx);
}

inline FloatType FloatType::getF16(MLIRContext *ctx) {
  return Float16Type::get(ctx);
}

inline FloatType FloatType::getTF32(MLIRContext *ctx) {
  return FloatTF32Type::get(ctx);
}

inline FloatType FloatType::getF32(MLIRContext *ctx) {
  return Float32Type::get(ctx);
}

inline FloatType FloatType::getF64(MLIRContext *ctx) {
  return Float64Type::get(ctx);
}

inline FloatType FloatType::getF80(MLIRContext *ctx) {
  return Float80Type::get(ctx);
}

inline FloatType FloatType::getF128(MLIRContext *ctx) {
  return Float128Type::get(ctx);
}

inline bool TensorType::classof(Type type) {
  return llvm::isa<RankedTensorType, UnrankedTensorType>(type);
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the strides of the MemRef if the layout map is in strided form.
/// MemRefs with a layout map in strided form include:
///   1. empty or identity layout map, in which case the stride information is
///      the canonical form computed from sizes;
///   2. a StridedLayoutAttr layout;
///   3. any other layout that be converted into a single affine map layout of
///      the form `K + k0 * d0 + ... kn * dn`, where K and ki's are constants or
///      symbols.
///
/// A stride specification is a list of integer values that are either static
/// or dynamic (encoded with ShapedType::kDynamic). Strides encode
/// the distance in the number of elements between successive entries along a
/// particular dimension.
LogicalResult getStridesAndOffset(MemRefType t,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);

/// Wrapper around getStridesAndOffset(MemRefType, SmallVectorImpl<int64_t>,
/// int64_t) that will assert if the logical result is not succeeded.
std::pair<SmallVector<int64_t>, int64_t> getStridesAndOffset(MemRefType t);

/// Return a version of `t` with identity layout if it can be determined
/// statically that the layout is the canonical contiguous strided layout.
/// Otherwise pass `t`'s layout into `simplifyAffineMap` and return a copy of
/// `t` with simplified layout.
MemRefType canonicalizeStridedLayout(MemRefType t);

/// Given MemRef `sizes` that are either static or dynamic, returns the
/// canonical "contiguous" strides AffineExpr. Strides are multiplicative and
/// once a dynamic dimension is encountered, all canonical strides become
/// dynamic and need to be encoded with a different symbol.
/// For canonical strides expressions, the offset is always 0 and the fastest
/// varying stride is always `1`.
///
/// Examples:
///   - memref<3x4x5xf32> has canonical stride expression
///         `20*exprs[0] + 5*exprs[1] + exprs[2]`.
///   - memref<3x?x5xf32> has canonical stride expression
///         `s0*exprs[0] + 5*exprs[1] + exprs[2]`.
///   - memref<3x4x?xf32> has canonical stride expression
///         `s1*exprs[0] + s0*exprs[1] + exprs[2]`.
AffineExpr makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                          ArrayRef<AffineExpr> exprs,
                                          MLIRContext *context);

/// Return the result of makeCanonicalStrudedLayoutExpr for the common case
/// where `exprs` is {d0, d1, .., d_(sizes.size()-1)}
AffineExpr makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                          MLIRContext *context);

/// Return "true" if the layout for `t` is compatible with strided semantics.
bool isStrided(MemRefType t);

/// Return "true" if the last dimension of the given type has a static unit
/// stride. Also return "true" for types with no strides.
bool isLastMemrefDimUnitStride(MemRefType type);

/// Return "true" if the last N dimensions of the given type are contiguous.
///
/// Examples:
///   - memref<5x4x3x2xi8, strided<[24, 6, 2, 1]> is contiguous when
///   considering both _all_ and _only_ the trailing 3 dims,
///   - memref<5x4x3x2xi8, strided<[48, 6, 2, 1]> is _only_ contiguous when
///   considering the trailing 3 dims.
///
bool trailingNDimsContiguous(MemRefType type, int64_t n);

} // namespace mlir

#endif // MLIR_IR_BUILTINTYPES_H
