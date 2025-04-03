//===- ReshapeOpsUtils.h - Utilities used by reshape ops --*- C++ -*------===//
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

#ifndef MLIR_DIALECT_UTILS_RESHAPEOPSUTILS_H
#define MLIR_DIALECT_UTILS_RESHAPEOPSUTILS_H

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {

using ReassociationIndices = SmallVector<int64_t, 2>;
using ReassociationIndicesRef = ArrayRef<int64_t>;
using ReassociationExprs = SmallVector<AffineExpr, 2>;

/// Attribute name for the ArrayAttr which encodes reassociation indices.
constexpr StringRef getReassociationAttrName() { return "reassociation"; }

/// Compose reassociation maps that are used in pair of reshape ops where one
/// is a producer and other is the consumer. Only valid to use this method when
/// both the producer and consumer are collapsing dimensions or both are
/// expanding dimensions.
///
/// For example,
///   producerReassociation = [[0, 1], [2], [3, 4]]
///   consumerReassociation = [[0, 1], [2]]
///
/// is folded into
///
///   result = [[0, 1, 2], [3, 4]].
std::optional<SmallVector<ReassociationIndices>> composeReassociationIndices(
    ArrayRef<ReassociationIndices> producerReassociations,
    ArrayRef<ReassociationIndices> consumerReassociations,
    MLIRContext *context);

/// Convert reassociation indices to affine expressions.
SmallVector<SmallVector<AffineExpr, 2>, 2> convertReassociationIndicesToExprs(
    MLIRContext *context, ArrayRef<ReassociationIndices> reassociationIndices);

/// Constructs affine maps out of Array<Array<AffineExpr>>.
SmallVector<AffineMap, 4>
getSymbolLessAffineMaps(ArrayRef<ReassociationExprs> reassociation);

/// Wraps a list of reassociations in an ArrayAttr.
ArrayAttr
getReassociationIndicesAttribute(OpBuilder &b,
                                 ArrayRef<ReassociationIndices> reassociation);

/// Convert Array<Array<AffineExpr>> to Array<Array<int64_t>>.
SmallVector<ReassociationIndices, 2> convertReassociationMapsToIndices(
    ArrayRef<ReassociationExprs> reassociationExprs);

/// Return the reassociations maps to use to reshape given the source type and
/// the target type when possible. Return std::nullopt when this computation
/// failed.
std::optional<SmallVector<ReassociationIndices>>
getReassociationIndicesForReshape(ShapedType sourceType, ShapedType targetType);

/// Returns the reassociation maps to collapse `sourceShape` to `targetShape` if
/// possible.
std::optional<SmallVector<ReassociationIndices>>
getReassociationIndicesForCollapse(ArrayRef<int64_t> sourceShape,
                                   ArrayRef<int64_t> targetShape);

/// Return true if the reassociation specification is valid, false otherwise.
/// When false, the `invalidIndex` integer pointer is optionally filled with the
/// index of the offending reassociation map.
bool isReassociationValid(ArrayRef<AffineMap> reassociation,
                          int *invalidIndex = nullptr);

template <typename ReshapeOpTy, typename InverseReshapeOpTy>
static OpFoldResult foldReshapeOp(ReshapeOpTy reshapeOp,
                                  ArrayRef<Attribute> operands) {
  // Fold identity reshape.
  if (reshapeOp.getSrcType() == reshapeOp.getType())
    return reshapeOp.getSrc();

  // Reshape of a constant can be replaced with a new constant.
  if (auto elements = dyn_cast_or_null<DenseElementsAttr>(operands.front()))
    return elements.reshape(cast<ShapedType>(reshapeOp.getResult().getType()));

  // Fold if the producer reshape source has the same shape with at most 1
  // dynamic dimension.
  auto reshapeSrcOp =
      reshapeOp.getSrc().template getDefiningOp<InverseReshapeOpTy>();
  if (!reshapeSrcOp)
    return nullptr;
  auto srcType = reshapeSrcOp.getSrcType();
  auto resultType = reshapeOp.getResultType();
  if (srcType != resultType)
    return nullptr;

  if (llvm::count_if(srcType.getShape(), ShapedType::isDynamic) < 2) {
    return reshapeSrcOp.getSrc();
  }

  // Fold producer-consumer reshape ops when they are perfect inverses of each
  // other:
  //   1) Reassociation indices are equivalent.
  //   2) Boundary types are equivalent.
  //   3) No reassociations have more than 1 dynamic dimension, and reassociated
  //      shapes are equal for each reassociation.
  auto reassociations = reshapeOp.getReassociationIndices();
  if (reassociations != reshapeSrcOp.getReassociationIndices())
    return nullptr;
  // If the reshapes are expanding and then collapsing, the ops can be folded
  // despite multiple dynamic dimensions.
  if (srcType.getRank() < reshapeSrcOp.getResultType().getRank())
    return reshapeSrcOp.getSrc();
  if (llvm::all_of(reassociations, [&](auto reInd) {
        ArrayRef<int64_t> srcSlice =
            srcType.getShape().slice(reInd.front(), reInd.size());
        return llvm::count_if(srcSlice, ShapedType::isDynamic) < 2;
      })) {
    return reshapeSrcOp.getSrc();
  }
  return nullptr;
}

/// Common verifier for reshape-like types. Fills `expandedType` and
///`collapsedType` with the proper `src` or `result` type.
template <typename Op, typename T>
static LogicalResult verifyReshapeLikeTypes(Op op, T expandedType,
                                            T collapsedType, bool isExpansion) {

  unsigned expandedRank = expandedType.getRank();
  unsigned collapsedRank = collapsedType.getRank();
  if (expandedRank < collapsedRank)
    return op.emitOpError("expected the expanded type, ")
           << expandedType << " to have a higher (or same) rank "
           << "than the collapsed type, " << collapsedType << '.';

  if (collapsedRank != op.getReassociation().size())
    return op.emitOpError("expected collapsed rank (")
           << collapsedRank << ") to equal the number of reassociation maps ("
           << op.getReassociation().size() << ").";

  auto maps = op.getReassociationMaps();
  for (auto it : llvm::enumerate(maps))
    if (it.value().getNumDims() != expandedRank)
      return op.emitOpError("expected reassociation map #")
             << it.index() << " to have size equal to the expanded rank ("
             << expandedRank << "), but it is  " << it.value().getNumDims()
             << '.';

  int invalidIdx = 0;
  if (!isReassociationValid(maps, &invalidIdx))
    return op.emitOpError("expected reassociation map #")
           << invalidIdx << " to be valid and contiguous.";

  return reshapeLikeShapesAreCompatible(
      [&](const Twine &msg) { return op->emitOpError(msg); },
      collapsedType.getShape(), expandedType.getShape(),
      op.getReassociationIndices(), isExpansion);
}

/// Verify that shapes of the reshaped types using following rule:
/// if a dimension in the collapsed type is static, then the corresponding
/// dimensions in the expanded shape should be
///    a) static
///    b) the product should be same as the collaped shape.
LogicalResult reshapeLikeShapesAreCompatible(
    function_ref<LogicalResult(const Twine &)> emitError,
    ArrayRef<int64_t> collapsedShape, ArrayRef<int64_t> expandedShape,
    ArrayRef<ReassociationIndices> reassociationMaps, bool isExpandingReshape);

/// Returns true iff the type is a MemRefType and has a non-identity layout.
bool hasNonIdentityLayout(Type type);

enum class ReshapeOpKind { kExpand, kCollapse };

/// Pattern to collapse producer/consumer reshape ops that are both collapsing
/// dimensions or are both expanding dimensions.
template <typename ReshapeOpTy, ReshapeOpKind opKind>
struct ComposeReassociativeReshapeOps : public OpRewritePattern<ReshapeOpTy> {
  using OpRewritePattern<ReshapeOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOpTy reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto srcReshapeOp =
        reshapeOp.getSrc().template getDefiningOp<ReshapeOpTy>();
    if (!srcReshapeOp)
      return failure();

    ShapedType resultType = reshapeOp.getResultType();

    if (hasNonIdentityLayout(srcReshapeOp.getSrc().getType()) ||
        hasNonIdentityLayout(reshapeOp.getSrc().getType()) ||
        hasNonIdentityLayout(reshapeOp.getResult().getType()))
      return failure();

    std::optional<SmallVector<ReassociationIndices>> reassociationIndices =
        composeReassociationIndices(srcReshapeOp.getReassociationIndices(),
                                    reshapeOp.getReassociationIndices(),
                                    rewriter.getContext());
    if (!reassociationIndices)
      return failure();

    if constexpr (opKind == ReshapeOpKind::kExpand) {
      SmallVector<OpFoldResult> outputShape(
          getMixedValues(reshapeOp.getStaticOutputShape(),
                         reshapeOp.getOutputShape(), rewriter));
      rewriter.replaceOpWithNewOp<ReshapeOpTy>(
          reshapeOp, resultType, srcReshapeOp.getSrc(), *reassociationIndices,
          outputShape);
    } else {
      rewriter.replaceOpWithNewOp<ReshapeOpTy>(
          reshapeOp, resultType, srcReshapeOp.getSrc(), *reassociationIndices);
    }
    return success();
  }
};

/// Pattern to compose
/// `collapse_shape(expand_shape(%src, reassociation_1), reassociation_2)`.
/// In that case both `srcType` and `resultType` can be expressed as a function
/// of `intermediateType`.
/// In order to demonstrate the approach, let's assume that `rank(srcType) >
/// `rank(resultType)`, i.e. the resulting operation should be `collapse_shape`.
/// In that case, we can iterate over every set of indices in `reassociation_2`
/// and try to find ids of sets of indices in `reassociation_1` that cover it
/// completely.
///
/// Example:
///
///   %0 = tensor.expand_shape %arg [[0], [1], [2, 3]]
///     : tensor<?x?x?xi64> into tensor<?x?x?x1xi64>
///   %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]]
///     : tensor<?x?x?x1xi64> into tensor<?x?xi64>
///
/// can be canonicalized into
///
///   %0 = tensor.collapse_shape %arg [[0, 1], [2]]
///     : tensor<?x?x?xi64> into tensor<?x?xi64>
///
/// because [0] and [1] from `expand_shape` reassociation cover completely
/// `[0, 1]` from `collapse_shape`. If it is impossible to find such union of
/// indices, then we fail.
//
/// When `rank(srcType) < rank(resultType)`, then we just swap `reassociation_1`
/// `reassociation_2` and produce `expand_shape`.
template <typename CollapseOpTy, typename ExpandOpTy, typename CastOpTy,
          typename DimOpTy, typename TensorTy>
struct ComposeCollapseOfExpandOp : public OpRewritePattern<CollapseOpTy> {
  using OpRewritePattern<CollapseOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(CollapseOpTy collapseOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = collapseOp.getSrc().template getDefiningOp<ExpandOpTy>();
    if (!expandOp)
      return failure();

    ShapedType srcType = expandOp.getSrcType();
    ShapedType resultType = collapseOp.getResultType();

    if (hasNonIdentityLayout(collapseOp.getSrc().getType()) ||
        hasNonIdentityLayout(expandOp.getSrc().getType()) ||
        hasNonIdentityLayout(expandOp.getResult().getType()))
      return failure();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();
    if (srcType == resultType)
      return failure();

    SmallVector<ReassociationIndices, 4> higherRankReassociation,
        lowerRankReassociation;

    if (srcRank > resultRank) {
      higherRankReassociation = expandOp.getReassociationIndices();
      lowerRankReassociation = collapseOp.getReassociationIndices();
    } else {
      higherRankReassociation = collapseOp.getReassociationIndices();
      lowerRankReassociation = expandOp.getReassociationIndices();
    }

    size_t higherRankIndicesID = 0;
    SmallVector<ReassociationIndices, 4> composedReassociation;
    for (const auto &lowerRankIndices : lowerRankReassociation) {
      ReassociationIndices composedIndices;
      while (higherRankIndicesID < higherRankReassociation.size()) {
        auto rightmostIndex =
            higherRankReassociation[higherRankIndicesID].back();
        if (rightmostIndex > lowerRankIndices.back())
          return failure();
        composedIndices.push_back(higherRankIndicesID++);
        if (rightmostIndex == lowerRankIndices.back())
          break;
      }
      composedReassociation.push_back(composedIndices);
    }
    if (srcRank > resultRank) {
      rewriter.replaceOpWithNewOp<CollapseOpTy>(
          collapseOp, resultType, expandOp.getSrc(), composedReassociation);
    } else if (srcRank < resultRank) {
      rewriter.replaceOpWithNewOp<ExpandOpTy>(
          collapseOp, resultType, expandOp.getSrc(), composedReassociation);
    } else {
      // Collapses/expansions that do not change the rank are not allowed. Use
      // a cast instead.
      assert(llvm::equal(srcType.getShape(), resultType.getShape()) &&
             "expected same shape");
      rewriter.replaceOpWithNewOp<CastOpTy>(collapseOp, resultType,
                                            expandOp.getSrc());
    }
    return success();
  }
};

template <typename ExpandOpTy, typename CollapseOpTy>
struct ComposeExpandOfCollapseOp : public OpRewritePattern<ExpandOpTy> {
  using OpRewritePattern<ExpandOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpandOpTy expandOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp = expandOp.getSrc().template getDefiningOp<CollapseOpTy>();
    if (!collapseOp)
      return failure();

    ShapedType srcType = collapseOp.getSrcType();
    ShapedType resultType = expandOp.getResultType();

    if (hasNonIdentityLayout(expandOp.getSrc().getType()) ||
        hasNonIdentityLayout(collapseOp.getSrc().getType()) ||
        hasNonIdentityLayout(collapseOp.getResult().getType()))
      return failure();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();
    if (srcType == resultType)
      return failure();

    auto srcReassociation = collapseOp.getReassociationIndices();
    auto resultReassociation = expandOp.getReassociationIndices();
    if (srcRank > resultRank) {
      auto composedReassociation = findCollapsingReassociation(
          srcReassociation, resultReassociation, srcType.getShape(),
          resultType.getShape());
      if (!composedReassociation)
        return failure();

      rewriter.replaceOpWithNewOp<CollapseOpTy>(
          expandOp, resultType, collapseOp.getSrc(), *composedReassociation);
      return success();
    }
    auto composedReassociation =
        findCollapsingReassociation(resultReassociation, srcReassociation,
                                    resultType.getShape(), srcType.getShape());
    if (!composedReassociation)
      return failure();

    SmallVector<OpFoldResult> outputShape(getMixedValues(
        expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter));
    rewriter.replaceOpWithNewOp<ExpandOpTy>(
        expandOp, resultType, collapseOp.getSrc(), *composedReassociation,
        outputShape);
    return success();
  }

private:
  // Attempts to find a way to collapse `srcShape` to `resultShape` by
  // collapsing subshapes defined by the reassociation indices.
  std::optional<SmallVector<ReassociationIndices>> findCollapsingReassociation(
      ArrayRef<ReassociationIndices> srcReassociation,
      ArrayRef<ReassociationIndices> resultReassociation,
      ArrayRef<int64_t> srcShape, ArrayRef<int64_t> resultShape) const {
    SmallVector<ReassociationIndices, 4> composedReassociation;

    if (srcReassociation.empty())
      return {getReassociationIndicesForCollapse(srcShape, resultShape)};

    for (auto item : llvm::zip(srcReassociation, resultReassociation)) {
      auto &srcIndices = std::get<0>(item);
      auto &resultIndices = std::get<1>(item);
      auto srcSubShape = srcShape.slice(srcIndices.front(), srcIndices.size());
      auto resultSubShape =
          resultShape.slice(resultIndices.front(), resultIndices.size());

      if (srcSubShape.size() == resultSubShape.size()) {
        if (srcSubShape == resultSubShape &&
            llvm::count_if(srcSubShape, ShapedType::isDynamic) < 2) {
          composedReassociation.push_back(srcIndices);
        } else {
          return std::nullopt;
        }
      }

      // Find reassociation to collapse `srcSubShape` into `resultSubShape`.
      auto subShapeReassociation =
          getReassociationIndicesForCollapse(srcSubShape, resultSubShape);
      if (!subShapeReassociation)
        return std::nullopt;

      // Remap the subshape indices back to the original srcShape.
      for (auto &subshape_indices : *subShapeReassociation) {
        ReassociationIndices shape_indices;
        for (int64_t index : subshape_indices)
          shape_indices.push_back(srcIndices.front() + index);
        composedReassociation.push_back(shape_indices);
      }
    }
    return {std::move(composedReassociation)};
  }
};

/// The input parameters `offsets`, `sizes`, `strides` specify a rectangular
/// non rank-reducing slice of the collapse_shape output. Try to find which
/// dimensions have been sliced and which dimensions are not sliced (offset = 0,
/// size = dim, size = 1). Note that this conservative as it cannot detect if a
/// dynamic size corresponds to the full tensor dimension or not.
llvm::SmallBitVector getSlicedDimensions(ArrayRef<OpFoldResult> sliceInputShape,
                                         ArrayRef<Range> sliceParams);

/// Determine which dimensions are linearized by a `tensor.collapse_shape` op by
/// inspecting its reassociation indices.
llvm::SmallBitVector
getLinearizedDimensions(ArrayRef<ReassociationIndices> reassociationIndices);

/// Given the parameters for both operations in a `CollapseShape->ExtractSlice`
/// chain and reified source and result shapes of the CollapseShapeOp, this
/// class provides two functions that assist with directly forming the result
/// of the extract slice by "tiling the CollapseShapeOp by 1".
//// Example:
// clang-format off
/// ```
/// %0 = linalg.generic ... -> tensor<3x7x11x10xf32>
/// %1 = tensor.collapse_shape %0 [[0, 1, 2], [3]] : ... to tensor<341x10xf32>
/// %2 = tensor.extract_slice %1 [13, 0] [10, 10] [2, 1] : .... tensor<10x10xf32>
/// ```
/// This class helps build the below IR to replace %2:
/// ```
/// %dest = tensor.empty() : tensor<10x10xf32>
/// %2 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg0) -> tensor<10x10xf32> {
///    %linear_index = affine.apply affine_map<(d0)[]->(d0*2 + 11)>(%iv)
///    %3:3 = arith.delinearize_index %iv into (3, 7, 11)
///
///    // This function takes %3 (multiIndices) and the parameters for the slice below.
///    %4 = tensor.extract_slice %0 [%3#0, %3#1, %3#2, 0] [1, 1, 1, 10] [1, 1, 1, 1] :
///          tensor<3x7x11x10xf32> to tensor<1x1x1x10xf32>
///
///    %5 = tensor.collapse_shape %4 [[0, 1, 2], [3]] : 
///          tensor<1x1x1x10xf32> into tensor<1x10xf32>
///    %6 = tensor.insert_slice %5 into %arg0 [%iv, 0] [1, 10] [1, 1] :
///          tensor<1x10xf32> into tensor<10x10xf32>
///    scf.yield %6 : tensor<10x10xf32>
/// }
/// ```
// clang-format on
class SliceFromCollapseHelper {
public:
  SliceFromCollapseHelper(ArrayRef<ReassociationIndices> reassociationIndices,
                          ArrayRef<OpFoldResult> collapseShapeInputShape,
                          ArrayRef<OpFoldResult> collapseShapeOutputShape,
                          ArrayRef<Range> extractSliceParams)
      : reassociationIndices(reassociationIndices),
        collapseShapeInputShape(collapseShapeInputShape),
        collapseShapeOutputShape(collapseShapeOutputShape),
        sliceParams(extractSliceParams),
        linearizedDimensions(getLinearizedDimensions(reassociationIndices)),
        slicedDimensions(getSlicedDimensions(collapseShapeOutputShape,
                                             extractSliceParams)) {}

  /// This function takes multi-indices and maps them to ExtractSlice parameters
  /// in the index space of the CollapseShape's source tensor. This function's
  /// signature can be described by `(D_0, D_1,.. D_{n-1}) -> (offsets, sizes,
  /// strides)` where `n` the number of "tiled dimensions", which are the
  /// dimensions of the output that are linearized by the collapse shape op and
  /// are also sliced. Each `D_i` is a tuple that must represent a valid
  /// multi-index for the `i-th` tiled dimension. In the example above, there is
  /// only one tiled dimension (D_0) and `arith.delinearize_index` produces the
  /// multi-index (%3) that would be passed to this function to generate the
  /// parameters for the `tensor.extract_slice` op (%4).
  SmallVector<Range> getExtractSliceParams(MLIRContext *ctx,
                                           ArrayRef<ValueRange> multiIndices);

  /// This function takes indices in the index space of the "tiled dimensions"
  /// described above and returns a set of Range variables that describe how the
  /// slice should be inserted into the destination. In the example above, `%iv`
  /// would be passed to this function to generate the parameters for the
  /// `tensor.insert_slice` op producing %6.
  SmallVector<Range> getInsertSliceParams(MLIRContext *ctx,
                                          ValueRange tileIndices);

private:
  SmallVector<ReassociationIndices> reassociationIndices;
  SmallVector<OpFoldResult> collapseShapeInputShape;
  SmallVector<OpFoldResult> collapseShapeOutputShape;
  SmallVector<Range> sliceParams;
  llvm::SmallBitVector linearizedDimensions;
  llvm::SmallBitVector slicedDimensions;
};

/// Parameters required to simplify a collapsing reshape op with a rank-reducing
/// slice operation. See `getSimplifyCollapseShapeWithRankReducingSliceInfo`.
struct CollapseShapeRankReducingSliceSimplificationInfo {
  /// The shape of the output of the rank-reducing slice.
  RankedTensorType sliceResultType;
  /// The reassociation indices for the new collapse shape op, if required. If
  /// `std::nullopt`, the slice should replace the collapse shape op.
  std::optional<SmallVector<ReassociationIndices>> newReassociationIndices;
};

/// A collapsing reshape operation can sometimes be simplified or eliminated by
/// inserting a single rank-reducing slice operation between it and the source
/// tensor. The slice op will either take the place of the source, allowing for
/// a new, simpler reshape op to replace the original, or the reshape op will be
/// completely replaced by the slice result.
///
/// This function returns the parameters required to implement this pattern. If
/// the pattern is not applicable, then failure is returned.
///
/// ### Example:
/// ```
/// %result = tensor.collapse_shape %0 [[0, 1], [2, 3]]
///    : tensor<?x1x30x10xf32> to tensor<?x300xf32>
/// ```
/// can be transformed to
/// ```
/// %tmp = tensor.extract_slice %0 [0, 0, 0, 0]
///                         [0, %dim1, 30, 30]
///                         [1, 1, 1 1]
///   : tensor<?x1x30x10xf32> to tensor<?x30x10xf32>
/// %result = tensor.collapse_shape %tmp [[0], [1, 2]]
///   : tensor<?x30x10xf32> to tensor<?x300xf32>
/// ```
///
/// ### Example:
/// ```
/// %result = tensor.collapse_shape %1 [[0, 1], [2]]
///    : tensor<?x1x30xf32> to tensor<?x30xf32>
/// ```
/// can be transformed to
/// ```
/// %result = tensor.extract_slice %1 [0, 0, 0]
///                                   [%dim2, 1, 30]
///                                   [1, 1, 1]
///    : tensor<?x1x30xf32> to tensor<?x30xf32>
/// ```
FailureOr<CollapseShapeRankReducingSliceSimplificationInfo>
getSimplifyCollapseShapeWithRankReducingSliceInfo(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices);

struct PackingMetadata {
  SmallVector<int64_t> insertPositions;
  SmallVector<int64_t> outerPositions;
  SmallVector<ReassociationIndices> reassociations;
};

/// Given a vector of `positions` indices representing desired packing insertion
/// points into a target vector (i.e. pack/unpack.inner_dim_pos), compute the
/// final positions in the target shape as well as the reshape reassociations.
// Note: This should not be called with a large positions array (or the
// implementation needs to be updated to use an N.log N sort instead of
// repeated N^2 counts).
PackingMetadata computePackingMetadata(int64_t packedRank,
                                       ArrayRef<int64_t> innerDimPos);
} // namespace mlir

#endif // MLIR_DIALECT_UTILS_RESHAPEOPSUTILS_H
