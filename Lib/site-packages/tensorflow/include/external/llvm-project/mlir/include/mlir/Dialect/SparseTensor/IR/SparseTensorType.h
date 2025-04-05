//===- SparseTensorType.h - Wrapper around RankedTensorType -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines the `SparseTensorType` wrapper class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORTYPE_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORTYPE_H_

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
/// A wrapper around `RankedTensorType`, which has three goals:
///
/// (1) To provide a uniform API for querying aspects of sparse-tensor
/// types; in particular, to make the "dimension" vs "level" distinction
/// overt (i.e., explicit everywhere).  Thus, throughout the sparsifier
/// this class should be preferred over using `RankedTensorType` or
/// `ShapedType` directly, since the methods of the latter do not make
/// the "dimension" vs "level" distinction overt.
///
/// (2) To provide a uniform abstraction over both sparse-tensor
/// types (i.e., `RankedTensorType` with `SparseTensorEncodingAttr`)
/// and dense-tensor types (i.e., `RankedTensorType` without an encoding).
/// That is, we want to manipulate dense-tensor types using the same API
/// that we use for manipulating sparse-tensor types; both to keep the
/// "dimension" vs "level" distinction overt, and to avoid needing to
/// handle certain cases specially in the sparsifier.
///
/// (3) To provide uniform handling of "defaults".  In particular
/// this means that dense-tensors should always return the same answers
/// as sparse-tensors with a default encoding.  But it additionally means
/// that the answers should be normalized, so that there's no way to
/// distinguish between non-provided data (which is filled in by default)
/// vs explicitly-provided data which equals the defaults.
///
class SparseTensorType {
public:
  // We memoize `lvlRank`, `dimToLvl`, and `lvlToDim` to avoid repeating
  // the conditionals throughout the rest of the class.
  SparseTensorType(RankedTensorType rtp)
      : rtp(rtp), enc(getSparseTensorEncoding(rtp)),
        lvlRank(enc ? enc.getLvlRank() : getDimRank()),
        dimToLvl(enc.isIdentity() ? AffineMap() : enc.getDimToLvl()),
        lvlToDim(enc.isIdentity() ? AffineMap() : enc.getLvlToDim()) {
    assert(rtp && "got null RankedTensorType");
    assert((!isIdentity() || getDimRank() == lvlRank) && "Rank mismatch");
  }

  SparseTensorType(ShapedType stp, SparseTensorEncodingAttr enc)
      : SparseTensorType(
            RankedTensorType::get(stp.getShape(), stp.getElementType(), enc)) {}

  SparseTensorType &operator=(const SparseTensorType &) = delete;
  SparseTensorType(const SparseTensorType &) = default;

  //
  // Factory methods to construct a new `SparseTensorType`
  // with the same dimension-shape and element type.
  //

  SparseTensorType withEncoding(SparseTensorEncodingAttr newEnc) const {
    return SparseTensorType(rtp, newEnc);
  }

  SparseTensorType withDimToLvl(AffineMap dimToLvl) const {
    return withEncoding(enc.withDimToLvl(dimToLvl));
  }

  SparseTensorType withDimToLvl(SparseTensorEncodingAttr dimToLvlEnc) const {
    return withEncoding(enc.withDimToLvl(dimToLvlEnc));
  }

  SparseTensorType withDimToLvl(const SparseTensorType &dimToLvlSTT) const {
    return withDimToLvl(dimToLvlSTT.getEncoding());
  }

  SparseTensorType withoutDimToLvl() const {
    return withEncoding(enc.withoutDimToLvl());
  }

  SparseTensorType withBitWidths(unsigned posWidth, unsigned crdWidth) const {
    return withEncoding(enc.withBitWidths(posWidth, crdWidth));
  }

  SparseTensorType withoutBitWidths() const {
    return withEncoding(enc.withoutBitWidths());
  }

  SparseTensorType withExplicitVal(Attribute explicitVal) const {
    return withEncoding(enc.withExplicitVal(explicitVal));
  }

  SparseTensorType withoutExplicitVal() const {
    return withEncoding(enc.withoutExplicitVal());
  }

  SparseTensorType withImplicitVal(Attribute implicitVal) const {
    return withEncoding(enc.withImplicitVal(implicitVal));
  }

  SparseTensorType withoutImplicitVal() const {
    return withEncoding(enc.withoutImplicitVal());
  }

  SparseTensorType
  withDimSlices(ArrayRef<SparseTensorDimSliceAttr> dimSlices) const {
    return withEncoding(enc.withDimSlices(dimSlices));
  }

  SparseTensorType withoutDimSlices() const {
    return withEncoding(enc.withoutDimSlices());
  }

  /// Allow implicit conversion to `RankedTensorType`, `ShapedType`,
  /// and `Type`.  These are implicit to help alleviate the impedance
  /// mismatch for code that has not been converted to use `SparseTensorType`
  /// directly.  Once more uses have been converted to `SparseTensorType`,
  /// we may want to make these explicit instead.
  ///
  /// WARNING: This user-defined-conversion method causes overload
  /// ambiguity whenever passing a `SparseTensorType` directly to a
  /// function which is overloaded to accept either `Type` or `TypeRange`.
  /// In particular, this includes `RewriterBase::replaceOpWithNewOp<OpTy>`
  /// and `OpBuilder::create<OpTy>` whenever the `OpTy::build` is overloaded
  /// thus.  This happens because the `TypeRange<T>(T&&)` ctor is implicit
  /// as well, and there's no SFINAE we can add to this method that would
  /// block subsequent application of that ctor.  The only way to fix the
  /// overload ambiguity is to avoid *implicit* conversion at the callsite:
  /// e.g., by using `static_cast` to make the conversion explicit, by
  /// assigning the `SparseTensorType` to a temporary variable of the
  /// desired type, etc.
  //
  // NOTE: We implement this as a single templated user-defined-conversion
  // function to avoid ambiguity problems when the desired result is `Type`
  // (since both `RankedTensorType` and `ShapedType` can be implicitly
  // converted to `Type`).
  template <typename T, typename = std::enable_if_t<
                            std::is_convertible_v<RankedTensorType, T>>>
  /*implicit*/ operator T() const {
    return rtp;
  }

  /// Explicitly convert to `RankedTensorType`.  This method is
  /// a convenience for resolving overload-ambiguity issues with
  /// implicit conversion.
  RankedTensorType getRankedTensorType() const { return rtp; }

  bool operator==(const SparseTensorType &other) const {
    // All other fields are derived from `rtp` and therefore don't need
    // to be checked.
    return rtp == other.rtp;
  }

  bool operator!=(const SparseTensorType &other) const {
    return !(*this == other);
  }

  MLIRContext *getContext() const { return rtp.getContext(); }

  Type getElementType() const { return rtp.getElementType(); }

  SparseTensorEncodingAttr getEncoding() const { return enc; }

  //
  // SparseTensorEncodingAttr delegators
  //

  /// Returns true for tensors which have an encoding, and false for
  /// those which do not.  Therefore tensors with an all-dense encoding
  /// return true.
  bool hasEncoding() const { return static_cast<bool>(enc); }

  /// Returns true for tensors where every level is dense.
  /// (This is always true for dense-tensors.)
  bool isAllDense() const { return enc.isAllDense(); }

  /// Returns true for tensors where every level is ordered.
  /// (This is always true for dense-tensors.)
  bool isAllOrdered() const { return enc.isAllOrdered(); }

  /// Translates between level / dimension coordinate space.
  ValueRange translateCrds(OpBuilder &builder, Location loc, ValueRange crds,
                           CrdTransDirectionKind dir) const {
    return enc.translateCrds(builder, loc, crds, dir);
  }

  /// Returns true if the dimToLvl mapping is a permutation.
  /// (This is always true for dense-tensors.)
  bool isPermutation() const { return enc.isPermutation(); }

  /// Returns true if the dimToLvl mapping is the identity.
  /// (This is always true for dense-tensors.)
  bool isIdentity() const { return enc.isIdentity(); }

  //
  // Other methods.
  //

  /// Returns the dimToLvl mapping (or the null-map for the identity).
  /// If you intend to compare the results of this method for equality,
  /// see `hasSameDimToLvl` instead.
  AffineMap getDimToLvl() const { return dimToLvl; }

  /// Returns the lvlToDiml mapping (or the null-map for the identity).
  AffineMap getLvlToDim() const { return lvlToDim; }

  /// Returns the dimToLvl mapping, where the identity map is expanded out
  /// into a full `AffineMap`.  This method is provided as a convenience,
  /// but for most purposes other methods (`isIdentity`, `getDimToLvl`,
  /// etc) will be more helpful.
  AffineMap getExpandedDimToLvl() const {
    return dimToLvl
               ? dimToLvl
               : AffineMap::getMultiDimIdentityMap(getDimRank(), getContext());
  }

  /// Returns true iff the two types have the same mapping.  This method
  /// takes care to handle identity maps properly, so it should be preferred
  /// over using `getDimToLvl` followed by `AffineMap::operator==`.
  bool hasSameDimToLvl(const SparseTensorType &other) const {
    // If the maps are the identity, then we need to check the rank
    // to be sure they're the same size identity.  (And since identity
    // means dimRank==lvlRank, we use lvlRank as a minor optimization.)
    return isIdentity() ? (other.isIdentity() && lvlRank == other.lvlRank)
                        : (dimToLvl == other.dimToLvl);
  }

  /// Returns the dimension-rank.
  Dimension getDimRank() const { return rtp.getRank(); }

  /// Returns the level-rank.
  Level getLvlRank() const { return lvlRank; }

  /// Returns the dimension-shape.
  ArrayRef<Size> getDimShape() const { return rtp.getShape(); }

  /// Returns the level-shape.
  SmallVector<Size> getLvlShape() const {
    return getEncoding().translateShape(getDimShape(),
                                        CrdTransDirectionKind::dim2lvl);
  }

  /// Returns the batched level-rank.
  unsigned getBatchLvlRank() const { return getEncoding().getBatchLvlRank(); }

  /// Returns the batched level-shape.
  SmallVector<Size> getBatchLvlShape() const {
    auto lvlShape = getEncoding().translateShape(
        getDimShape(), CrdTransDirectionKind::dim2lvl);
    lvlShape.truncate(getEncoding().getBatchLvlRank());
    return lvlShape;
  }

  /// Returns the type with an identity mapping.
  RankedTensorType getDemappedType() const {
    return RankedTensorType::get(getLvlShape(), getElementType(),
                                 enc.withoutDimToLvl());
  }

  /// Safely looks up the requested dimension-DynSize.  If you intend
  /// to check the result with `ShapedType::isDynamic`, then see the
  /// `getStaticDimSize` method instead.
  Size getDynamicDimSize(Dimension d) const {
    assert(d < getDimRank() && "Dimension is out of bounds");
    return getDimShape()[d];
  }

  /// Returns true if no dimension has dynamic size.
  bool hasStaticDimShape() const { return rtp.hasStaticShape(); }

  /// Returns true if any dimension has dynamic size.
  bool hasDynamicDimShape() const { return !hasStaticDimShape(); }

  /// Returns true if the given dimension has dynamic size.  If you
  /// intend to call `getDynamicDimSize` based on the result, then see
  /// the `getStaticDimSize` method instead.
  bool isDynamicDim(Dimension d) const {
    // We don't use `rtp.isDynamicDim(d)` because we want the
    // OOB error message to be consistent with `getDynamicDimSize`.
    return ShapedType::isDynamic(getDynamicDimSize(d));
  }

  /// Returns the number of dimensions which have dynamic sizes.
  /// The return type is `int64_t` to maintain consistency with
  /// `ShapedType::Trait<T>::getNumDynamicDims`.
  int64_t getNumDynamicDims() const { return rtp.getNumDynamicDims(); }

  ArrayRef<LevelType> getLvlTypes() const { return enc.getLvlTypes(); }
  LevelType getLvlType(Level l) const {
    // This OOB check is for dense-tensors, since this class knows
    // their lvlRank (whereas STEA::getLvlType will/can only check
    // OOB for sparse-tensors).
    assert(l < lvlRank && "Level out of bounds");
    return enc.getLvlType(l);
  }

  // We can't just delegate these, since we want to use this class's
  // `getLvlType` method instead of STEA's.
  bool isDenseLvl(Level l) const { return isDenseLT(getLvlType(l)); }
  bool isCompressedLvl(Level l) const { return isCompressedLT(getLvlType(l)); }
  bool isLooseCompressedLvl(Level l) const {
    return isLooseCompressedLT(getLvlType(l));
  }
  bool isSingletonLvl(Level l) const { return isSingletonLT(getLvlType(l)); }
  bool isNOutOfMLvl(Level l) const { return isNOutOfMLT(getLvlType(l)); }
  bool isOrderedLvl(Level l) const { return isOrderedLT(getLvlType(l)); }
  bool isUniqueLvl(Level l) const { return isUniqueLT(getLvlType(l)); }
  bool isWithPos(Level l) const { return isWithPosLT(getLvlType(l)); }
  bool isWithCrd(Level l) const { return isWithCrdLT(getLvlType(l)); }

  /// Returns the coordinate-overhead bitwidth, defaulting to zero.
  unsigned getCrdWidth() const { return enc ? enc.getCrdWidth() : 0; }

  /// Returns the position-overhead bitwidth, defaulting to zero.
  unsigned getPosWidth() const { return enc ? enc.getPosWidth() : 0; }

  /// Returns the explicit value, defaulting to null Attribute for unset.
  Attribute getExplicitVal() const {
    return enc ? enc.getExplicitVal() : nullptr;
  }

  /// Returns the implicit value, defaulting to null Attribute for 0.
  Attribute getImplicitVal() const {
    return enc ? enc.getImplicitVal() : nullptr;
  }

  /// Returns the coordinate-overhead MLIR type, defaulting to `IndexType`.
  Type getCrdType() const { return enc.getCrdElemType(); }

  /// Returns the position-overhead MLIR type, defaulting to `IndexType`.
  Type getPosType() const { return enc.getPosElemType(); }

  /// Returns true iff this sparse tensor type has a trailing
  /// COO region starting at the given level. By default, it
  /// tests for a unique COO type at top level.
  bool isCOOType(Level startLvl = 0, bool isUnique = true) const;

  /// Returns the starting level of this sparse tensor type for a
  /// trailing COO region that spans **at least** two levels. If
  /// no such COO region is found, then returns the level-rank.
  ///
  /// DEPRECATED: use getCOOSegment instead;
  Level getAoSCOOStart() const { return getEncoding().getAoSCOOStart(); };

  /// Returns [un]ordered COO type for this sparse tensor type.
  RankedTensorType getCOOType(bool ordered) const;

  /// Returns a list of COO segments in the sparse tensor types.
  SmallVector<COOSegment> getCOOSegments() const {
    return getEncoding().getCOOSegments();
  }

private:
  // These two must be const, to ensure coherence of the memoized fields.
  const RankedTensorType rtp;
  const SparseTensorEncodingAttr enc;
  // Memoized to avoid frequent redundant conditionals.
  const Level lvlRank;
  const AffineMap dimToLvl;
  const AffineMap lvlToDim;
};

/// Convenience methods to obtain a SparseTensorType from a Value.
inline SparseTensorType getSparseTensorType(Value val) {
  return SparseTensorType(cast<RankedTensorType>(val.getType()));
}
inline std::optional<SparseTensorType> tryGetSparseTensorType(Value val) {
  if (auto rtp = dyn_cast<RankedTensorType>(val.getType()))
    return SparseTensorType(rtp);
  return std::nullopt;
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORTYPE_H_
