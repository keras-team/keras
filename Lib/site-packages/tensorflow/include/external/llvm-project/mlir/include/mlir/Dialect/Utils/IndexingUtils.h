//===- IndexingUtils.h - Helpers related to index computations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and common canonicalization patterns for
// reshape operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_INDEXINGUTILS_H
#define MLIR_DIALECT_UTILS_INDEXINGUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include <optional>
#include <utility>

namespace mlir {
class ArrayAttr;

//===----------------------------------------------------------------------===//
// Utils that operate on static integer values.
//===----------------------------------------------------------------------===//

/// Given a set of sizes, return the suffix product.
///
/// When applied to slicing, this is the calculation needed to derive the
/// strides (i.e. the number of linear indices to skip along the (k-1) most
/// minor dimensions to get the next k-slice).
///
/// This is the basis to linearize an n-D offset confined to `[0 ... sizes]`.
///
/// Assuming `sizes` is `[s0, .. sn]`, return the vector<int64_t>
///   `[s1 * ... * sn, s2 * ... * sn, ..., sn, 1]`.
///
/// `sizes` elements are asserted to be non-negative.
///
/// Return an empty vector if `sizes` is empty.
SmallVector<int64_t> computeSuffixProduct(ArrayRef<int64_t> sizes);
inline SmallVector<int64_t> computeStrides(ArrayRef<int64_t> sizes) {
  return computeSuffixProduct(sizes);
}

/// Return a vector containing llvm::zip_equal(v1, v2) multiplied elementwise.
///
/// Return an empty vector if `v1` and `v2` are empty.
SmallVector<int64_t> computeElementwiseMul(ArrayRef<int64_t> v1,
                                           ArrayRef<int64_t> v2);

/// Self-explicit.
int64_t computeSum(ArrayRef<int64_t> basis);

/// Self-explicit.
int64_t computeProduct(ArrayRef<int64_t> basis);

/// Return the number of elements of basis (i.e. the max linear index).
/// Return `0` if `basis` is empty.
///
/// `basis` elements are asserted to be non-negative.
///
/// Return `0` if `basis` is empty.
inline int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  return computeProduct(basis);
}

/// Return the linearized index of 'offsets' w.r.t. 'basis'.
///
/// `basis` elements are asserted to be non-negative.
int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension space,
/// return the vector-space offsets in each dimension for a de-linearized index.
/// `strides` elements are asserted to be non-negative.
///
/// Let `li = linearIndex`, assuming `strides` are `[s0, .. sn]`, return the
/// vector of int64_t
///   `[li % s0, (li / s0) % s1, ..., (li / s0 / .. / sn-1) % sn]`
SmallVector<int64_t> delinearize(int64_t linearIndex,
                                 ArrayRef<int64_t> strides);

/// Return the multi-dimensional integral ratio of `subShape` to the trailing
/// dimensions of `shape`. This represents how many times `subShape` fits
/// within `shape`. If integral division is not possible, return std::nullopt.
/// The trailing `subShape.size()` entries of both shapes are assumed (and
/// enforced) to only contain non-negative values.
///
/// Examples:
///   - shapeRatio({3, 5, 8}, {2, 5, 2}) returns {3, 2, 1}.
///   - shapeRatio({3, 8}, {2, 5, 2}) returns std::nullopt (subshape has
///   higher
///     rank).
///   - shapeRatio({42, 2, 10, 32}, {2, 5, 2}) returns {42, 1, 2, 16} which is
///     derived as {42(leading shape dim), 2/2, 10/5, 32/2}.
///   - shapeRatio({42, 2, 11, 32}, {2, 5, 2}) returns std::nullopt  which is
///     derived as {42(leading shape dim), 2/2, 11/5(not divisible), 32/2}.
std::optional<SmallVector<int64_t>>
computeShapeRatio(ArrayRef<int64_t> shape, ArrayRef<int64_t> subShape);

//===----------------------------------------------------------------------===//
// Utils that operate on AffineExpr.
//===----------------------------------------------------------------------===//

/// Given a set of sizes, return the suffix product.
///
/// When applied to slicing, this is the calculation needed to derive the
/// strides (i.e. the number of linear indices to skip along the (k-1) most
/// minor dimensions to get the next k-slice).
///
/// This is the basis to linearize an n-D offset confined to `[0 ... sizes]`.
///
/// Assuming `sizes` is `[s0, .. sn]`, return the vector<AffineExpr>
///   `[s1 * ... * sn, s2 * ... * sn, ..., sn, 1]`.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that
/// result in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide
/// by an AffineDimExpr).
///
/// `sizes` elements are expected to bind to non-negative values.
///
/// Return an empty vector if `sizes` is empty.
SmallVector<AffineExpr> computeSuffixProduct(ArrayRef<AffineExpr> sizes);
inline SmallVector<AffineExpr> computeStrides(ArrayRef<AffineExpr> sizes) {
  return computeSuffixProduct(sizes);
}

/// Return a vector containing llvm::zip_equal(v1, v2) multiplied elementwise.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that
/// result in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide
/// by an AffineDimExpr).
///
/// Return an empty vector if `v1` and `v2` are empty.
SmallVector<AffineExpr> computeElementwiseMul(ArrayRef<AffineExpr> v1,
                                              ArrayRef<AffineExpr> v2);

/// Self-explicit.
AffineExpr computeSum(MLIRContext *ctx, ArrayRef<AffineExpr> basis);

/// Self-explicit.
AffineExpr computeProduct(MLIRContext *ctx, ArrayRef<AffineExpr> basis);

/// Return the number of elements of basis (i.e. the max linear index).
/// Return `0` if `basis` is empty.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that
/// result in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide
/// by an AffineDimExpr).
///
/// `basis` elements are expected to bind to non-negative values.
///
/// Return the `0` AffineConstantExpr if `basis` is empty.
inline AffineExpr computeMaxLinearIndex(MLIRContext *ctx,
                                        ArrayRef<AffineExpr> basis) {
  return computeProduct(ctx, basis);
}

/// Return the linearized index of 'offsets' w.r.t. 'basis'.
///
/// Assuming `offsets` is `[o0, .. on]` and `basis` is `[b0, .. bn]`, return the
/// AffineExpr `o0 * b0 + .. + on * bn`.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that result
/// in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide by an
/// AffineDimExpr).
///
/// `basis` elements are expected to bind to non-negative values.
AffineExpr linearize(MLIRContext *ctx, ArrayRef<AffineExpr> offsets,
                     ArrayRef<AffineExpr> basis);
AffineExpr linearize(MLIRContext *ctx, ArrayRef<AffineExpr> offsets,
                     ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension space,
/// return the vector-space offsets in each dimension for a de-linearized index.
///
/// Let `li = linearIndex`, assuming `strides` are `[s0, .. sn]`, return the
/// vector of AffineExpr
///   `[li % s0, (li / s0) % s1, ..., (li / s0 / .. / sn-1) % sn]`
///
/// It is the caller's responsibility to pass proper AffineExpr kind that result
/// in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide by an
/// AffineDimExpr).
///
/// `strides` elements are expected to bind to non-negative values.
SmallVector<AffineExpr> delinearize(AffineExpr linearIndex,
                                    ArrayRef<AffineExpr> strides);
SmallVector<AffineExpr> delinearize(AffineExpr linearIndex,
                                    ArrayRef<int64_t> strides);

//===----------------------------------------------------------------------===//
// Permutation utils.
//===----------------------------------------------------------------------===//

template <typename T>
SmallVector<T> applyPermutation(ArrayRef<T> input,
                                ArrayRef<int64_t> permutation) {
  assert(input.size() == permutation.size() &&
         "expected input rank to equal permutation rank");
  assert(
      llvm::all_of(permutation, [&](size_t s) { return s < input.size(); }) &&
      "permutation must be within input bounds");
  auto permutationRange = llvm::map_range(
      llvm::seq<unsigned>(0, input.size()),
      [&](int64_t idx) -> T { return input[permutation[idx]]; });
  return llvm::to_vector(permutationRange);
}

template <typename T>
SmallVector<T> applyPermutation(const SmallVectorImpl<T> &input,
                                ArrayRef<int64_t> permutation) {
  return applyPermutation(ArrayRef(input), permutation);
}

/// Apply the permutation defined by `permutation` to `inVec`.
/// Element `i` in `inVec` is mapped to location `j = permutation[i]`.
/// E.g.: for an input vector `inVec = ['a', 'b', 'c']` and a permutation
/// vector `permutation = [2, 0, 1]`, this function leaves `inVec = ['c', 'a',
/// 'b']`.
template <typename T, unsigned N>
void applyPermutationToVector(SmallVector<T, N> &inVec,
                              ArrayRef<int64_t> permutation) {
  inVec = applyPermutation(inVec, permutation);
}

/// Helper method to apply to inverse a permutation.
SmallVector<int64_t> invertPermutationVector(ArrayRef<int64_t> permutation);

/// Returns true if `permutation` is an identity permutation.
bool isIdentityPermutation(ArrayRef<int64_t> permutation);

/// Method to check if an interchange vector is a permutation.
bool isPermutationVector(ArrayRef<int64_t> interchange);

/// Return a permutation vector of size permSize that would result in moving
/// positions into desiredPositions.
///
/// For example, permSize == 5, positions = {2, 4}, desiredPositions = {1, 0}
/// would result in a {4, 2, 0, 1, 3} permutation vector.
SmallVector<int64_t>
computePermutationVector(int64_t permSize, ArrayRef<int64_t> positions,
                         ArrayRef<int64_t> desiredPositions);

/// Returns a permutation vector that drop the input dims in
/// dropPositions from inputPerm.
///
/// For example, inputPerm = {2, 4, 0, 1, 3} and dropPositions= {1, 2} would
/// result in a {2, 0, 1} permutation vector.
SmallVector<int64_t> dropDims(ArrayRef<int64_t> inputPerm,
                              ArrayRef<int64_t> dropPositions);

/// Helper to return a subset of `arrayAttr` as a vector of int64_t.
// TODO: Port everything relevant to DenseArrayAttr and drop this util.
SmallVector<int64_t> getI64SubArray(ArrayAttr arrayAttr, unsigned dropFront = 0,
                                    unsigned dropBack = 0);

/// Compute linear index from provided strides and indices, assuming strided
/// layout.
/// Returns AffineExpr and list of values to apply to it, e.g.:
///
/// auto &&[expr, values] = computeLinearIndex(...);
/// offset = affine::makeComposedFoldedAffineApply(builder, loc, expr, values);
std::pair<AffineExpr, SmallVector<OpFoldResult>>
computeLinearIndex(OpFoldResult sourceOffset, ArrayRef<OpFoldResult> strides,
                   ArrayRef<OpFoldResult> indices);
std::pair<AffineExpr, SmallVector<OpFoldResult>>
computeLinearIndex(OpFoldResult sourceOffset, ArrayRef<int64_t> strides,
                   ArrayRef<Value> indices);

//===----------------------------------------------------------------------===//
// Utilities for decomposing larger shapes
//===----------------------------------------------------------------------===//

namespace detail {
/// Encapsulates the set of parameters that are used to make tile offset
/// calculations in the TileOffsetRangeIterator.
class TileOffsetRangeImpl {
public:
  TileOffsetRangeImpl(ArrayRef<int64_t> shape, ArrayRef<int64_t> tileShape,
                      ArrayRef<int64_t> loopOrder);

  int64_t getMaxLinearIndex() const { return maxLinearIndex; }

  SmallVector<int64_t> getStaticTileOffsets(int64_t linearIndex) const;

  SmallVector<AffineExpr> getDynamicTileOffsets(AffineExpr linearIndex) const;

  template <typename T>
  SmallVector<T> getTileOffsets(T linearIndex) const {
    if constexpr (std::is_same_v<T, int64_t>)
      return getStaticTileOffsets(linearIndex);
    else
      return getDynamicTileOffsets(linearIndex);
  }

  size_t getRank() const { return tileShape.size(); }

private:
  /// The sub-shape that divides the larger outer shape (which is provided to
  /// the constructor).
  SmallVector<int64_t> tileShape;
  /// The inverse permutation to the `loopOrder` permutation provided in the
  /// constructor.
  SmallVector<int64_t> inverseLoopOrder;
  /// The strides for the basis 'div(shape, tileShape)' permuted by `loopOrder`.
  SmallVector<int64_t> sliceStrides;
  /// The maximum linear index in the iteration space given by basis 'div(shape,
  /// tileShape)'.
  int64_t maxLinearIndex;
};

/// The STL-style iterator implementation for StaticTileOffsetRange.
template <typename ElementType>
class TileOffsetRangeIterator
    : public llvm::iterator_facade_base<TileOffsetRangeIterator<ElementType>,
                                        std::forward_iterator_tag,
                                        SmallVector<ElementType>> {
public:
  TileOffsetRangeIterator(const TileOffsetRangeImpl &params, ElementType index)
      : params(params), index(index) {}

  void operator++() { incrementIndex(1); }
  TileOffsetRangeIterator operator++(int) {
    const auto copy = *this;
    ++*this;
    return copy;
  }

  bool operator==(const TileOffsetRangeIterator &other) const {
    return index == other.index;
  }
  bool operator!=(const TileOffsetRangeIterator &other) const {
    return index != other.index;
  }

  SmallVector<ElementType> operator*() const {
    return params.getTileOffsets(index);
  }
  void operator+=(int64_t offset) { incrementIndex(offset); }

private:
  void incrementIndex(int64_t offset) { index = index + offset; }
  const TileOffsetRangeImpl params;
  int64_t index;
};
} // namespace detail

/// A range-style iterator that allows for iterating over the offsets of all
/// potential tiles of size `tileShape` within the larger shape `shape`, using
/// an ordering specified by `loopOrder`. The `loopOrder` specifies the order of
/// unrolling by numbering the dimensions in order from "outer most for loop"
/// (slowest changing) to "inner most for loop" (fastest changing).
///
/// For example, for `shape = {10, 20, 30}`, `tileShape = {5, 10, 15}`, and
/// `loopOrder={2, 0, 1}`, the iterating over this range will yield offsets:
///
/// ```
/// {0, 0,  0}, {0, 10,  0}, {5, 0,  0}, {5, 10,  0}, {0, 0, 15},
/// {0, 10, 15}, {5, 0, 15}, {0, 10, 15}, {5, 10, 15}
/// ```
///
/// This is useful in contexts where a vector computation over a larger shape
/// needs to be unrolled to a set of operations on subsets of the original
/// operands, such as during the "vector unrolling" transformations.
///
/// The size of `tileShape` must be less-than-or-equal-to the size of `shape`.a
/// If the rank of `tileShape` is smaller than `shape`, then `tileShape`
/// elements correspond to the trailing dimensions of `shape`, and the leading
/// dimensions are considered untiled and `tileShape` is effectively prepended
/// with the leading dims of `shape`.
class StaticTileOffsetRange {
public:
  using IteratorTy = detail::TileOffsetRangeIterator<int64_t>;
  using ParamsTy = detail::TileOffsetRangeImpl;

  StaticTileOffsetRange(ArrayRef<int64_t> shape, ArrayRef<int64_t> tileShape,
                        ArrayRef<int64_t> loopOrder)
      : params(shape, tileShape, loopOrder), beginValue(params, 0),
        pastEndValue(params, params.getMaxLinearIndex()) {
    assert(shape.size() >= tileShape.size());
    assert(loopOrder.size() == shape.size());
  }

  /// Create the range with identity loop order.
  StaticTileOffsetRange(ArrayRef<int64_t> shape, ArrayRef<int64_t> tileShape)
      : params(shape, tileShape,
               llvm::to_vector(llvm::seq<int64_t>(0, shape.size()))),
        beginValue(params, 0),
        pastEndValue(params, params.getMaxLinearIndex()) {
    assert(shape.size() >= tileShape.size());
  }

  IteratorTy begin() const { return beginValue; }
  IteratorTy end() const { return pastEndValue; }

  /// Returns the total number of tiles that fit in the larger shape.
  size_t size() const { return params.getMaxLinearIndex(); }

  /// Returns rank of the iterator's shape.
  size_t getRank() const { return params.getRank(); }

private:
  const ParamsTy params;
  IteratorTy beginValue;
  IteratorTy pastEndValue;
};
} // namespace mlir

#endif // MLIR_DIALECT_UTILS_INDEXINGUTILS_H
