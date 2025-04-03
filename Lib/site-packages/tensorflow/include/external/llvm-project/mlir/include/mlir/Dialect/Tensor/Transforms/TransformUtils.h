//===- TransformsUtils.h - Tensor Transformation Utilities-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMUTILS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMUTILS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// Extract slice from `tensor.collapse_shape`
//===----------------------------------------------------------------------===//

/// This class assists with generating IR required to materialize an
/// arbitrary-sized slice from the result of a CollapseShapeOp. In order to
/// accomplish this, a loop nest or similar operation must be created by the
/// caller. The purpose of the loop nest is to generate a "tiling by 1" of all
/// sliced dimensions. The "tiling by 1" assembles all elements of the result
/// tile over dimensions that would have been impossible to directly slice.
///
/// The class provides three methods:
/// 1. `ExtractSliceFromCollapseHelper::create`: emits IR that should
/// appear before the loop nest and populates the internal state.
/// 2. `ExtractSliceFromCollapseHelper::getIterationSpaceSizes`: returns
/// parameters used by the caller to construct the loop nest.
/// 3. `ExtractSliceFromCollapseHelper::emitLoopNestBody`:
/// emits IR to construct a "size-1 tile" of the desired result and returns a
/// set of ranges where the tile should be inserted into the destination
/// tensor.
///
/// ### Intended usage:
///
/// The caller should first call `ExtractSliceFromCollapseHelper::create` and
/// then create a destination tensor that is the same size as the desired slice.
/// The caller then creates a loop nest that iterates over the multi-dimensional
/// iteration space defined by `[0, ub[0]) x [0, ub[1]] x ... x [0, ub[N-1]]`
/// where `ub` is the upper bound given by
/// `ExtractSliceFromCollapseHelper::getIterationSpaceSizes`. Inside the body of
/// the loop nest, the caller should call
/// `ExtractSliceFromCollapseHelper::emitLoopNestBody` and provide the induction
/// variables. This returns a sub-tile and a set of ranges that describe where
/// this tile should be inserted into the result by the caller. For a complete
/// example of usage, see the examples in the TestTensorTransforms pass.
///
/// ### Example:
/// Consider the following IR:
/// ```
/// %0 = linalg.generic ... -> tensor<3x?x?x11x?xf32>
/// %1 = tensor.collapse_shape %0 [[0, 1, 2], [3, 4]]
///        : tensor<3x?x?x11x?xf32> into tensor<?x?xf32>
/// %2 = tensor.extract_slice %1 [%offt0, %offt1][%size0, %size1][1, 1]
///        : tensor<?x?xf32> to tensor<?x?xf32>
/// ```
///
/// We can construct %2 by generating the following, which only uses `%0`:
///
/// ```
/// %dest = tensor.empty(%size0, %size1) : tensor<?x?xf32>
/// %1 = tensor.dim %0, %c1 : tensor<3x?x?x11x?xf32>
/// %2 = tensor.dim %0, %c2 : tensor<3x?x?x11x?xf32>
/// %3 = tensor.dim %0, %c4 : tensor<3x?x?x11x?xf32>
///
/// %result = scf.for %iv0 = %c0 to %arg2 step %c1 iter_args(%arg6 = %dest) ->
///                                                  (tensor<?x?xf32>) {
///   %5 = scf.for %iv1 = %c0 to %arg4 step %c1 iter_args(%arg8 = %arg6)
///                                                  -> (tensor<?x?xf32>) {
///     %lin0 = (affine.apply) %iv0 + %offt0
///     %lin1 = (affine.apply) %iv1 + %offt1
///
///     %mi0:3 = affine.delinearize_index %lin0 into (%c3, %1, %2)
///     %mi1:2 = affine.delinearize_index %lin1 into (%c11, %3)
///
///     %sub_tile = tensor.extract_slice %0
///                    [%mi0#0, %mi0#1, %mi0#2, %mi1#0, %mi1#1]
///                    [1, 1, 1, 1, 1]
///                    [1, 1, 1, 1, 1]
///            : tensor<3x?x?x11x?xf32> to tensor<1x1x1x1x1xf32>
///     %sub_tile_collapsed = tensor.collapse_shape %sub_tile
///             [[0, 1, 2], [3, 4]]
///            : tensor<1x1x1x1x1xf32> into tensor<1x1xf3
///
///     %12 = tensor.insert_slice %sub_tile_collapsed into
///             %arg8[%iv0, %iv1] [1, 1] [1, 1]
///             : tensor<1x1xf32> into tensor<?x?xf32>
///     scf.yield %12 : tensor<?x?xf32>
///   }
///   scf.yield %5 : tensor<?x?xf32>
/// }
/// ```
///
/// ### Explanation of example:
///
/// Each step above is explained below.
///
/// #### Step 0: Create %dest and materialization of shapes.
/// This step is self-explanatory and performed by the caller. It can be
/// done before or after calling `ExtractSliceFromCollapseHelper::create`,
/// which materializes the source shape (`%0, %1, %2`).
///
/// #### Step 1: Create loop nest.
///
/// The caller creates the loop nest (depicted here is `scf.for`, but any other
/// similar op can be used). The iteration should start at zero and proceed with
/// step size 1 to the upper bounds given by
/// `ExtractSliceFromCollapseHelper::getIterationSpaceSizes`. This forms the
/// basis for the "tiling by 1".
///
/// #### Step 2: Transform (%iv0, %iv1) from the index space of %3 to the index
/// space of %0.
///
/// This step is performed by
/// `ExtractSliceFromCollapseHelper::emitLoopNestBody`.
///
/// The induction variables `%iv0` and `%iv1` live in the
/// index space of %2 (for dimensions 0 and 1, respectively). `%lin0` and
/// `%lin1` are the result of inverting or resolve the index space
/// transformation represented by the slice operation, accounting for offset and
/// stride. Subsequently, `%mi0` and `%mi1` are the result of applying the
/// inverse index space transformation represented by `tensor.collapse_shape`.
/// This is accomplished using `affine.delinearize_index`. Note that %iv0
/// and %iv1 now correspond to multi-indices `%mi0:3` and `%mi1:2`.
///
/// #### Step 3: Extract a sub-tile slice from the source.
///
/// This step is also performed by
/// `ExtractSliceFromCollapseHelper::emitLoopNestBody`.
///
/// The indices `%mi0` and `%mi1` are used to extract a slice from %0.  This
/// slice is then collapsed down to match the result rank.
///
/// #### Step 4: Insert sub-tile into the destination
///
/// This step is performed by the caller using the results of
/// `ExtractSliceFromCollapseHelper::emitLoopNestBody`.
///
/// In the above example, the slice insertion parameters are straightforward,
/// but in other possible situations, the slice parameters are more complicated,
/// which is why this helper calculates them for the caller. These other
/// situations correspond to:
/// 1. The presence of linearized dimensions that are not sliced
/// 2. The presence of non-linearized dimensions that are sliced.
class ExtractSliceFromCollapseHelper {
public:
  /// Given a CollapseShapeOp and a set of ranges describing the desired slice
  /// of its result, emits IR to materialize the shapes of the input and output
  /// tensors, and returns an instance of the initialized class. Returns failure
  /// if the slice is rank-reducing.
  static FailureOr<ExtractSliceFromCollapseHelper>
  create(OpBuilder &b, tensor::CollapseShapeOp op, ArrayRef<Range> sliceParams);

  /// Given a CollapseShapeOp and an ExtractSliceOp acting on its result, emits
  /// IR to materialize the shapes of the input and output tensors of the
  /// CollapseShapeOp, and returns an instance of the initialized class. Returns
  /// failure if the slice is rank-reducing.
  static FailureOr<ExtractSliceFromCollapseHelper>
  create(OpBuilder &b, tensor::CollapseShapeOp collapseOp,
         tensor::ExtractSliceOp extractOp);

  ExtractSliceFromCollapseHelper(
      tensor::CollapseShapeOp collapseShapeOp,
      ArrayRef<OpFoldResult> collapseShapeInputShape,
      ArrayRef<OpFoldResult> collapseShapeOutputShape,
      ArrayRef<Range> extractSliceParams,
      const llvm::SmallBitVector &linearizedDimensions,
      const llvm::SmallBitVector &slicedDimensions, ArrayRef<Value> tiledSizes)
      : collapseShapeOp(collapseShapeOp),
        collapseShapeInputShape(collapseShapeInputShape),
        collapseShapeOutputShape(collapseShapeOutputShape),
        sliceParams(extractSliceParams),
        linearizedDimensions(linearizedDimensions),
        slicedDimensions(slicedDimensions), tiledSizes(tiledSizes) {}

  /// Return the upper bounds of the iteration space (with 0 offset and stride
  /// 1) required to create the desired slice. Note that this is not the same
  /// as the `sizes` parameters of the ExtractSliceOp because not all dimensions
  /// of the slice are required to be tiled to form the result.
  const SmallVector<Value> &getIterationSpaceSizes() { return tiledSizes; }

  /// Generates the IR inside of the caller's loop nest for 1) inverting the
  /// index mappings of the ExtractSliceOp->CollapseShapeOp chain and 2)
  /// extracting the CollapseShapeOp source tensor tile for this specified
  /// iteration space point `tileInductionVars` and 3) calculating where to
  /// insert the extracted tile. The returned pair consists of the results of
  /// (2) and (3) and should be used by the caller to insert into the
  /// destination tensor.
  std::pair<Value, SmallVector<Range>>
  emitLoopNestBody(OpBuilder &builder, Location loc,
                   ValueRange tileInductionVars);

private:
  tensor::CollapseShapeOp collapseShapeOp;
  SmallVector<OpFoldResult> collapseShapeInputShape;
  SmallVector<OpFoldResult> collapseShapeOutputShape;
  SmallVector<Range> sliceParams;
  llvm::SmallBitVector linearizedDimensions;
  llvm::SmallBitVector slicedDimensions;
  SmallVector<Value> tiledSizes;
};

/// Tries to simplify a `tensor.collapse_shape` operation by inserting a single
/// rank-reducing `tensor.extract_slice` operation. The `extract_slice` op will
/// either take the place of the source, allowing for a new, simpler
/// `collapse_shape` op to replace `op`, or the `collapse_shape` op will be
/// completely replaced by the `extract_slice` result. Either way, `op` is
/// replaced and the new op is returned.
///
/// ### Example:
/// ```
/// %result = tensor.collapse_shape %0 [[0, 1], [2, 3]]
///    : tensor<?x1x30x10xf32> to tensor<?x300xf32>
/// ```
/// can be transformed to
///
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
///
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
///
/// ### Unsupported cases:
///
/// This transform doesn't yet support reducing the rank of the reassociation
/// indices, which would require inserting a `tensor.expand_shape` op similar to
/// the following example:
/// ```
/// %result = tensor.collapse_shape %0 [[0, 1], [2, 3]]
///    : tensor<1x1x30x10xf32> to tensor<1x300xf32>
/// ```
/// can be transformed to
/// ```
/// %tmp = tensor.extract_slice %0 [0, 0, 0, 0]
///                         [0, 1, 30, 30]
///                         [1, 1, 1 1]
///   : tensor<1x1x30x10xf32> to tensor<30x10xf32>
/// %result0 = tensor.collapse_shape %tmp [[0, 1]]
///   : tensor<30x10xf32> to tensor<300xf32>
/// %result1 = tensor.expand_shape %tmp [[0, 1], [2]] :... tensor<1x300xf32>
/// ```
///
FailureOr<Operation *>
simplifyCollapseShapeWithRankReducingExtractSlice(tensor::CollapseShapeOp op,
                                                  RewriterBase &rewriter);
} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMUTILS_H
