//===- LinalgInterface.h - Linalg operations interfaces -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for Linalg operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_
#define MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/RawOstreamExtras.h"

namespace mlir {
namespace linalg {
class IteratorTypeAttr;
class LinalgOp;
class GenericOp;

namespace detail {
/// Implementation of the method that check if given operands
/// can be dropped, i.e. the remaining operands can compute the loop
/// bounds of the op.
bool canOpOperandsBeDroppedImpl(linalg::LinalgOp linalgOp,
                                ArrayRef<OpOperand *> droppedOperands);
} // namespace detail

/// Positions of a Linalg op loops that correspond to different kinds of a
/// contraction dimension.
struct ContractionDimensions {
  SmallVector<unsigned, 2> batch;
  SmallVector<unsigned, 2> m;
  SmallVector<unsigned, 2> n;
  SmallVector<unsigned, 2> k;
};

/// Find at least 2 parallel (m and n) and 1 reduction (k) dimension candidates
/// that form a matmul subcomputation within `linalgOp`.
/// These dimensions are such that:
///   1. The m dimension is involved in an outer-product along LHS
///      (i.e. it is a permutation on RES and LHS and does not appear in RHS).
///   2. The n dimension is involved in an outer-product along RHS
///      (i.e. it is a permutation on RES and RHS and does not appear in LHS).
///   3. The k dimension appears as a permutation on LHS and RHS.
///   4. m, n and k appear only once in any given indexing.
///   5. Optional batch dimensions that appear in all operands are captured.
/// This allows e.g. detecting that some contraction is embedded within
/// `linalgOp` with some orthogonal heuristic.
/// When multiple dimension occurrences exist that match `batch`, `m`, `n`, or
/// `k`, indices are returned in sorted order.
/// Returns a failure if any of `m`, `n` or `k` is empty.
FailureOr<ContractionDimensions> inferContractionDims(LinalgOp linalgOp);
FailureOr<ContractionDimensions>
inferContractionDims(ArrayRef<AffineMap> indexingMaps);

/// Checks whether `linalgOp` conforms to ContractionOpInterface.
// TODO: embed within `isa<ContractionOpInterface>` if possible / natural.
bool isaContractionOpInterface(LinalgOp linalgOp);

/// Positions of a Linalg op loops that correspond to different kinds of a
/// convolution dimension.
struct ConvolutionDimensions {
  SmallVector<unsigned, 2> batch;
  SmallVector<unsigned, 2> outputImage;
  SmallVector<unsigned, 2> outputChannel;
  SmallVector<unsigned, 2> filterLoop;
  SmallVector<unsigned, 2> inputChannel;
  SmallVector<unsigned, 2> depth;
  SmallVector<int64_t, 2> strides;
  SmallVector<int64_t, 2> dilations;
};

/// Find at least 1 parallel (output_image) and reduction (filter_loop)
/// dimension candidates that form a convolution subcomputation within
/// `linalgOp`. The LHS is assumed to be the convolution input while the
/// RHS is assumed as the filter.
/// These dimensions are such that:
///   1. Optional batch dimensions that appear in the input and filter.
///   2. The output_image dimension is involved in a cross-correlation along LHS
///      (i.e. it is a permutation on RES and LHS and has an associated
///      filter_loop in RHS).
///   3. Optional output_channel dimension is involved in an outer-product along
///      RHS (i.e. it is a permutation on RES and RHS and does not appear in
///      LHS).
///   4. Optional input_channel dimension appears as a permutation on LHS and
///      RHS.
///   5. The filter_loop dimension appears as a permutation on the RHS and
///      represents the shape of the kernel cross-correlated along a
///      corresponding output_image dim.
///   6. The input_channel dimension appears as a permutation on LHS and RHS.
///   7. All dimensions appear only once in any given indexing map.
/// This allows e.g. detecting that some convolution is embedded within
/// `linalgOp` with some orthogonal heuristic.
/// When multiple dimension occurrences exist that match any classification
/// indices are returned in sorted order.
/// Returns a failure if `output_image` (and implicitly `filter_loop`) is empty.
FailureOr<ConvolutionDimensions> inferConvolutionDims(LinalgOp linalgOp);

/// Checks whether `linalgOp` conforms to ConvolutionOpInterface.
/// By default, we require the `linalgOp` to have non-empty convolved dims
/// (implicitly non-empty `output_image` and `filter_loop`).
/// Users can loosen the constraint by setting `allowEmptyConvolvedDims` to true
// TODO: embed within `isa<ConvolutionOpInterface>` if possible / natural.
bool isaConvolutionOpInterface(LinalgOp linalgOp,
                               bool allowEmptyConvolvedDims = false);

/// Checks whether `linalgOp` is semantically equivalent to a `linalg.copyOp`.
bool isaCopyOpInterface(LinalgOp linalgOp);

/// Checks whether a given `genericOp` is semantically equivalent to a single
/// linalgelementwise unary op. e.g. linalg.exp.
/// A linalg.generic body could be a series of unary elementwise ops e.g.
/// `exp(neg(x))`, such as formed by linalg op fusion. Here we restrict it to
/// detecting cases where body is is a single computation op.
bool isaElemwiseSingleUnaryOpInterface(GenericOp genericOp);

/// Checks whether `genericOp` is semantically equivalent to a single linalg
/// elementwise binary op e.g. linalg.sub.
bool isaElemwiseSingleBinaryOpInterface(GenericOp genericOp);

/// Checks whether `genericOp` is semantically equivalent to a `linalg.fill`.
/// Returns the scalar fill value if true.
std::optional<Value> isaFillOpInterface(GenericOp genericOp);

namespace detail {

/// Returns true if the block contains a contraction of the following form:
///
///   %0 = <elemwise>(permutation-of(cu(block-argument-0),
///                                  cu(block-argument-1)))
///   %1 = <reduce>(permutation-of(cu(%0), cu(block-argument-2)))
///   return-like cu(%1)
///
/// where <elemwise> and <reduce> are binary operations constituting a
/// contraction (in the canonical case, <elemwise> is a multiplication and
/// <reduce> is an addition). The name and other properties of these operations
/// are checked by `isaPair`. All operands of all operations may be supplied
/// through a chain of side effect-free unary operations, such as casts, which
/// is denoted as `cu` above.
///
/// When the body does not contain a contraction, a more precise description of
/// the failed precondition is send to the `errs` stream, if provided.
bool isContractionBody(Block &block,
                       function_ref<bool(Operation *, Operation *)> isaPair,
                       llvm::raw_ostream &errs = mlir::thread_safe_nulls());

/// Result of matching a Linalg generic against the predicates of it being a
/// contraction.
enum class MatchContractionResult;

/// Checks whether `op` conforms to ContractionOpInterface and populates
/// `dimensions` with indexes of the different kinds of dimensions when
/// present.
MatchContractionResult
isContractionInterfaceImpl(Operation *op,
                           ContractionDimensions *dimensions = nullptr);

/// Returns the error message corresponding to the contraction checking return
/// code.
StringRef getMatchContractionMessage(MatchContractionResult res);

/// Result of matching a Linalg generic against the predicates of it being a
/// convolution.
enum class MatchConvolutionResult;

/// Checks whether `op` conforms to ConvolutionOpInterface and populates
/// `dimensions` with indexes of the different kinds of dimensions when
/// present.
/// If `allowEmptyConvolvedDims` is not set, we further checks whether the `op`
/// contains convolved dims.
MatchConvolutionResult
isConvolutionInterfaceImpl(Operation *op,
                           ConvolutionDimensions *dimensions = nullptr,
                           bool allowEmptyConvolvedDims = false);

/// Returns the error message corresponding to the convolution checking return
/// code.
StringRef getMatchConvolutionMessage(MatchConvolutionResult res);

/// Verify that `op` conforms to ContractionOpInterface.
LogicalResult verifyContractionInterface(Operation *op);

/// Verify that `op` conforms to the ConvolutionOpInterface.
LogicalResult verifyConvolutionInterface(Operation *op);

/// Verify that `op` conforms to the FillOpInterface.
LogicalResult verifyFillInterface(Operation *op);

/// Verify that `op` conforms to the invariants of StructuredOpInterface
LogicalResult verifyStructuredOpInterface(Operation *op);

} // namespace detail
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

/// Include the generated interface declarations.
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_
