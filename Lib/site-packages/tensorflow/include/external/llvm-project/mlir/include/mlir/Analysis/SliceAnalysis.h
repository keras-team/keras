//===- SliceAnalysis.h - Analysis for Transitive UseDef chains --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_SLICEANALYSIS_H_
#define MLIR_ANALYSIS_SLICEANALYSIS_H_

#include <functional>
#include <vector>

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
class BlockArgument;
class Operation;
class Value;

struct SliceOptions {
  /// Type of the condition to limit the propagation of transitive use-defs.
  /// This can be used in particular to limit the propagation to a given Scope
  /// or to avoid passing through certain types of operation in a configurable
  /// manner.
  using TransitiveFilter = std::function<bool(Operation *)>;
  TransitiveFilter filter = nullptr;

  /// Include the top level op in the slice.
  bool inclusive = false;

  // TODO: Remove this alias once downstream users are updated.
  SliceOptions() {}
  SliceOptions(TransitiveFilter filter) : filter(std::move(filter)) {}
};

// TODO: Remove this alias once downstream users are updated.
using TransitiveFilter = SliceOptions::TransitiveFilter;

struct BackwardSliceOptions : public SliceOptions {
  using SliceOptions::SliceOptions;
  /// When omitBlockArguments is true, the backward slice computation omits
  /// traversing any block arguments. When omitBlockArguments is false, the
  /// backward slice computation traverses block arguments and asserts that the
  /// parent op has a single region with a single block.
  bool omitBlockArguments = false;
};

using ForwardSliceOptions = SliceOptions;

/// Fills `forwardSlice` with the computed forward slice (i.e. all
/// the transitive uses of op), **without** including that operation.
///
/// This additionally takes a TransitiveFilter which acts as a frontier:
/// when looking at uses transitively, an operation that does not pass the
/// filter is never propagated through. This allows in particular to carve out
/// the scope within a ForOp or the scope within an IfOp.
///
/// The implementation traverses the use chains in postorder traversal for
/// efficiency reasons: if an operation is already in `forwardSlice`, no
/// need to traverse its uses again. Since use-def chains form a DAG, this
/// terminates.
///
/// Upon return to the root call, `forwardSlice` is filled with a
/// postorder list of uses (i.e. a reverse topological order). To get a proper
/// topological order, we just reverse the order in `forwardSlice` before
/// returning.
///
/// Example starting from node 0
/// ============================
///
///               0
///    ___________|___________
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
/// 1. after getting back to the root getForwardSlice, `forwardSlice` may
///    contain:
///      {9, 7, 8, 5, 1, 2, 6, 3, 4}
/// 2. reversing the result of 1. gives:
///      {4, 3, 6, 2, 1, 5, 8, 7, 9}
///
void getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});

/// Value-rooted version of `getForwardSlice`. Return the union of all forward
/// slices for the uses of the value `root`.
void getForwardSlice(Value root, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});

/// Fills `backwardSlice` with the computed backward slice (i.e.
/// all the transitive defs of op), **without** including that operation.
///
/// This additionally takes a TransitiveFilter which acts as a frontier:
/// when looking at defs transitively, an operation that does not pass the
/// filter is never propagated through. This allows in particular to carve out
/// the scope within a ForOp or the scope within an IfOp.
///
/// The implementation traverses the def chains in postorder traversal for
/// efficiency reasons: if an operation is already in `backwardSlice`, no
/// need to traverse its definitions again. Since useuse-def chains form a DAG,
/// this terminates.
///
/// Upon return to the root call, `backwardSlice` is filled with a
/// postorder list of defs. This happens to be a topological order, from the
/// point of view of the use-def chains.
///
/// Example starting from node 8
/// ============================
///
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
///    {1, 2, 5, 3, 4, 6}
///
void getBackwardSlice(Operation *op, SetVector<Operation *> *backwardSlice,
                      const BackwardSliceOptions &options = {});

/// Value-rooted version of `getBackwardSlice`. Return the union of all backward
/// slices for the op defining or owning the value `root`.
void getBackwardSlice(Value root, SetVector<Operation *> *backwardSlice,
                      const BackwardSliceOptions &options = {});

/// Iteratively computes backward slices and forward slices until
/// a fixed point is reached. Returns an `SetVector<Operation *>` which
/// **includes** the original operation.
///
/// This allows building a slice (i.e. multi-root DAG where everything
/// that is reachable from an Value in forward and backward direction is
/// contained in the slice).
/// This is the abstraction we need to materialize all the operations for
/// supervectorization without worrying about orderings and Value
/// replacements.
///
/// Example starting from any node
/// ==============================
///
///    1       2      3      4
///    |_______|      |______|
///    |   |             |   |
///    |   5             6___|
///    |___|_____________|   |
///      |               |   |
///      7               8   |
///      |_______________|   |
///              |           |
///              9          10
///
/// Return the whole DAG in some topological order.
///
/// The implementation works by just filling up a worklist with iterative
/// alternate calls to `getBackwardSlice` and `getForwardSlice`.
///
/// The following section describes some additional implementation
/// considerations for a potentially more efficient implementation but they are
/// just an intuition without proof, we still use a worklist for now.
///
/// Additional implementation considerations
/// ========================================
/// Consider the defs-op-uses hourglass.
///    ____
///    \  /  defs (in some topological order)
///     \/
///     op
///     /\
///    /  \  uses (in some topological order)
///   /____\
///
/// We want to iteratively apply `getSlice` to construct the whole
/// list of Operation that are reachable by (use|def)+ from op.
/// We want the resulting slice in topological order.
/// Ideally we would like the ordering to be maintained in-place to avoid
/// copying Operation at each step. Keeping this ordering by construction
/// seems very unclear, so we list invariants in the hope of seeing whether
/// useful properties pop up.
///
/// In the following:
///   we use |= for set inclusion;
///   we use << for set topological ordering (i.e. each pair is ordered).
///
/// Assumption:
/// ===========
/// We wish to maintain the following property by a recursive argument:
///   """
///      defs << {op} <<uses are in topological order.
///   """
/// The property clearly holds for 0 and 1-sized uses and defs;
///
/// Invariants:
///   2. defs and uses are in topological order internally, by construction;
///   3. for any {x} |= defs, defs(x) |= defs;    because all go through op
///   4. for any {x} |= uses,    defs |= defs(x); because all go through op
///   5. for any {x} |= defs,    uses |= uses(x); because all go through op
///   6. for any {x} |= uses, uses(x) |= uses;    because all go through op
///
/// Intuitively, we should be able to recurse like:
///   preorder(defs) - op - postorder(uses)
/// and keep things ordered but this is still hand-wavy and not worth the
/// trouble for now: punt to a simple worklist-based solution.
///
SetVector<Operation *>
getSlice(Operation *op, const BackwardSliceOptions &backwardSliceOptions = {},
         const ForwardSliceOptions &forwardSliceOptions = {});

/// Utility to match a generic reduction given a list of iteration-carried
/// arguments, `iterCarriedArgs` and the position of the potential reduction
/// argument within the list, `redPos`. If a reduction is matched, returns the
/// reduced value and the topologically-sorted list of combiner operations
/// involved in the reduction. Otherwise, returns a null value.
///
/// The matching algorithm relies on the following invariants, which are subject
/// to change:
///  1. The first combiner operation must be a binary operation with the
///     iteration-carried value and the reduced value as operands.
///  2. The iteration-carried value and combiner operations must be side
///     effect-free, have single result and a single use.
///  3. Combiner operations must be immediately nested in the region op
///     performing the reduction.
///  4. Reduction def-use chain must end in a terminator op that yields the
///     next iteration/output values in the same order as the iteration-carried
///     values in `iterCarriedArgs`.
///  5. `iterCarriedArgs` must contain all the iteration-carried/output values
///     of the region op performing the reduction.
///
/// This utility is generic enough to detect reductions involving multiple
/// combiner operations (disabled for now) across multiple dialects, including
/// Linalg, Affine and SCF. For the sake of genericity, it does not return
/// specific enum values for the combiner operations since its goal is also
/// matching reductions without pre-defined semantics in core MLIR. It's up to
/// each client to make sense out of the list of combiner operations. It's also
/// up to each client to check for additional invariants on the expected
/// reductions not covered by this generic matching.
Value matchReduction(ArrayRef<BlockArgument> iterCarriedArgs, unsigned redPos,
                     SmallVectorImpl<Operation *> &combinerOps);

} // namespace mlir

#endif // MLIR_ANALYSIS_SLICEANALYSIS_H_
