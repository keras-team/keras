//===- TopologicalSortUtils.h - Topological sort utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_TOPOLOGICALSORTUTILS_H
#define MLIR_ANALYSIS_TOPOLOGICALSORTUTILS_H

#include "mlir/IR/Block.h"

namespace mlir {

/// Given a block, sort a range operations in said block in topological order.
/// The main purpose is readability of graph regions, potentially faster
/// processing of certain transformations and analyses, or fixing the SSA
/// dominance of blocks that require it after transformations. The function
/// sorts the given operations such that, as much as possible, all users appear
/// after their producers.
///
/// For example:
///
/// ```mlir
/// %0 = test.foo
/// %1 = test.bar %0, %2
/// %2 = test.baz
/// ```
///
/// Will become:
///
/// ```mlir
/// %0 = test.foo
/// %1 = test.baz
/// %2 = test.bar %0, %1
/// ```
///
/// The sort also works on operations with regions and implicit captures. For
/// example:
///
/// ```mlir
/// %0 = test.foo {
///   test.baz %1
///   %1 = test.bar %2
/// }
/// %2 = test.foo
/// ```
///
/// Will become:
///
/// ```mlir
/// %0 = test.foo
/// %1 = test.foo {
///   test.baz %2
///   %2 = test.bar %0
/// }
/// ```
///
/// Note that the sort is not recursive on nested regions. This sort is stable;
/// if the operations are already topologically sorted, nothing changes.
///
/// Operations that form cycles are moved to the end of the block in order. If
/// the sort is left with only operations that form a cycle, it breaks the cycle
/// by marking the first encountered operation as ready and moving on.
///
/// The function optionally accepts a callback that can be provided by users to
/// virtually break cycles early. It is called on top-level operations in the
/// block with value uses at or below those operations. The function should
/// return true to mark that value as ready to be scheduled.
///
/// For example, if `isOperandReady` is set to always mark edges from `foo.A` to
/// `foo.B` as ready, these operations:
///
/// ```mlir
/// %0 = foo.B(%1)
/// %1 = foo.C(%2)
/// %2 = foo.A(%0)
/// ```
///
/// Are sorted as:
///
/// ```mlir
/// %0 = foo.A(%2)
/// %1 = foo.C(%0)
/// %2 = foo.B(%1)
/// ```
bool sortTopologically(
    Block *block, iterator_range<Block::iterator> ops,
    function_ref<bool(Value, Operation *)> isOperandReady = nullptr);

/// Given a block, sort its operations in topological order, excluding its
/// terminator if it has one. This sort is stable.
bool sortTopologically(
    Block *block,
    function_ref<bool(Value, Operation *)> isOperandReady = nullptr);

/// Compute a topological ordering of the given ops. This sort is not stable.
///
/// Note: If the specified ops contain incomplete/interrupted SSA use-def
/// chains, the result may not actually be a topological sorting with respect to
/// the entire program.
bool computeTopologicalSorting(
    MutableArrayRef<Operation *> ops,
    function_ref<bool(Value, Operation *)> isOperandReady = nullptr);

/// Gets a list of blocks that is sorted according to dominance. This sort is
/// stable.
SetVector<Block *> getBlocksSortedByDominance(Region &region);

/// Sorts all operations in `toSort` topologically while also considering region
/// semantics. Does not support multi-sets.
SetVector<Operation *> topologicalSort(const SetVector<Operation *> &toSort);

} // end namespace mlir

#endif // MLIR_ANALYSIS_TOPOLOGICALSORTUTILS_H
