//===- LoopInvariantCodeMotionUtils.h - LICM Utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOPINVARIANTCODEMOTIONUTILS_H
#define MLIR_TRANSFORMS_LOOPINVARIANTCODEMOTIONUTILS_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {

class LoopLikeOpInterface;
class Operation;
class Region;
class RewriterBase;
class Value;

/// Given a list of regions, perform loop-invariant code motion. An operation is
/// loop-invariant if it depends only of values defined outside of the loop.
/// LICM moves these operations out of the loop body so that they are not
/// computed more than once.
///
/// Example:
///
/// ```mlir
/// affine.for %arg0 = 0 to 10 {
///   affine.for %arg1 = 0 to 10 {
///     %v0 = arith.addi %arg0, %arg0 : i32
///     %v1 = arith.addi %v0, %arg1 : i32
///   }
/// }
/// ```
///
/// After LICM:
///
/// ```mlir
/// affine.for %arg0 = 0 to 10 {
///   %v0 = arith.addi %arg0, %arg0 : i32
///   affine.for %arg1 = 0 to 10 {
///     %v1 = arith.addi %v0, %arg1 : i32
///   }
/// }
/// ```
///
/// Users must supply three callbacks.
///
/// - `isDefinedOutsideRegion` returns true if the given value is invariant with
///   respect to the given region. A common implementation might be:
///   `value.getParentRegion()->isProperAncestor(region)`.
/// - `shouldMoveOutOfRegion` returns true if the provided operation can be
///   moved of the given region, e.g. if it is side-effect free.
/// - `moveOutOfRegion` moves the operation out of the given region. A common
///   implementation might be: `op->moveBefore(region->getParentOp())`.
///
/// An operation is moved if all of its operands satisfy
/// `isDefinedOutsideRegion` and it satisfies `shouldMoveOutOfRegion`.
///
/// Returns the number of operations moved.
size_t moveLoopInvariantCode(
    ArrayRef<Region *> regions,
    function_ref<bool(Value, Region *)> isDefinedOutsideRegion,
    function_ref<bool(Operation *, Region *)> shouldMoveOutOfRegion,
    function_ref<void(Operation *, Region *)> moveOutOfRegion);

/// Move side-effect free loop invariant code out of a loop-like op using
/// methods provided by the interface.
size_t moveLoopInvariantCode(LoopLikeOpInterface loopLike);

/// Hoist loop-invariant tensor subsets (subset extraction and subset insertion
/// ops) from loop-like ops. Extraction ops are moved before the loop. Insertion
/// ops are moved after the loop. The loop body operates on newly added region
/// iter_args (one per extraction-insertion pair).
///
/// A subset extraction op (`SubsetExtractionOpInterface`) extracts from a
/// tensor value at a subset. The result of the op may have an arbitrary type,
/// i.e., not necessarily a tensor type. Example: "tensor.extract_slice".
///
/// A subset insertion op  (`SubsetInsertionOpInterface`) inserts into a tensor
/// value ("destination") at a subset. Example: "tensor.insert_slice".
///
/// Matching extraction-insertion subset ops can be hoisted from a loop if there
/// are no other ops within the loop that operate on the same or on an
/// overlapping subset. In particular, non-subset ops can prevent hoisting
/// because the analysis does not know what subset they operate on.
///
/// Example:
/// ```
/// %r = scf.for ... iter_args(%t = %a) -> (tensor<?xf32>) {
///   %0 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
///   %1 = "test.foo"(%0) : (tensor<5xf32>) -> (tensor<5xf32>)
///   %2 = tensor.insert_slice %1 into %t[0][5][1]
///       : tensor<5xf32> into tensor<?xf32>
///   scf.yield %2 : tensor<?xf32>
/// }
/// ```
/// Is rewritten to:
/// ```
/// %0 = tensor.extract_slice %a[0][5][1] : tensor<?xf32> to tensor<5xf32>
/// %new_loop:2 = scf.for ... iter_args(%t = %a, %h = %0) -> (tensor<?xf32>) {
///   %1 = "test.foo"(%h) : (tensor<5xf32>) -> (tensor<5xf32>)
///   scf.yield %t, %2 : tensor<?xf32>, tensor<5xf32>
/// }
/// %r = tensor.insert_slice %new_loop#1 into %new_loop#0
///     : tensor<5xf32> into tensor<?xf32>
/// ```
LoopLikeOpInterface hoistLoopInvariantSubsets(RewriterBase &rewriter,
                                              LoopLikeOpInterface loopLike);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOPINVARIANTCODEMOTIONUTILS_H
