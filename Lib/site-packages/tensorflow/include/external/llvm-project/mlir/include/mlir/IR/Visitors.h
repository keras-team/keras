//===- Visitors.h - Utilities for visiting operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for walking and visiting operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VISITORS_H
#define MLIR_IR_VISITORS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class Diagnostic;
class InFlightDiagnostic;
class Operation;
class Block;
class Region;

/// A utility result that is used to signal how to proceed with an ongoing walk:
///   * Interrupt: the walk will be interrupted and no more operations, regions
///   or blocks will be visited.
///   * Advance: the walk will continue.
///   * Skip: the walk of the current operation, region or block and their
///   nested elements that haven't been visited already will be skipped and will
///   continue with the next operation, region or block.
class WalkResult {
  enum ResultEnum { Interrupt, Advance, Skip } result;

public:
  WalkResult(ResultEnum result = Advance) : result(result) {}

  /// Allow LogicalResult to interrupt the walk on failure.
  WalkResult(LogicalResult result)
      : result(failed(result) ? Interrupt : Advance) {}

  /// Allow diagnostics to interrupt the walk.
  WalkResult(Diagnostic &&) : result(Interrupt) {}
  WalkResult(InFlightDiagnostic &&) : result(Interrupt) {}

  bool operator==(const WalkResult &rhs) const { return result == rhs.result; }
  bool operator!=(const WalkResult &rhs) const { return result != rhs.result; }

  static WalkResult interrupt() { return {Interrupt}; }
  static WalkResult advance() { return {Advance}; }
  static WalkResult skip() { return {Skip}; }

  /// Returns true if the walk was interrupted.
  bool wasInterrupted() const { return result == Interrupt; }

  /// Returns true if the walk was skipped.
  bool wasSkipped() const { return result == Skip; }
};

/// Traversal order for region, block and operation walk utilities.
enum class WalkOrder { PreOrder, PostOrder };

/// This iterator enumerates the elements in "forward" order.
struct ForwardIterator {
  /// Make operations iterable: return the list of regions.
  static MutableArrayRef<Region> makeIterable(Operation &range);

  /// Regions and block are already iterable.
  template <typename T>
  static constexpr T &makeIterable(T &range) {
    return range;
  }
};

/// A utility class to encode the current walk stage for "generic" walkers.
/// When walking an operation, we can either choose a Pre/Post order walker
/// which invokes the callback on an operation before/after all its attached
/// regions have been visited, or choose a "generic" walker where the callback
/// is invoked on the operation N+1 times where N is the number of regions
/// attached to that operation. The `WalkStage` class below encodes the current
/// stage of the walk, i.e., which regions have already been visited, and the
/// callback accepts an additional argument for the current stage. Such
/// generic walkers that accept stage-aware callbacks are only applicable when
/// the callback operates on an operation (i.e., not applicable for callbacks
/// on Blocks or Regions).
class WalkStage {
public:
  explicit WalkStage(Operation *op);

  /// Return true if parent operation is being visited before all regions.
  bool isBeforeAllRegions() const { return nextRegion == 0; }
  /// Returns true if parent operation is being visited just before visiting
  /// region number `region`.
  bool isBeforeRegion(int region) const { return nextRegion == region; }
  /// Returns true if parent operation is being visited just after visiting
  /// region number `region`.
  bool isAfterRegion(int region) const { return nextRegion == region + 1; }
  /// Return true if parent operation is being visited after all regions.
  bool isAfterAllRegions() const { return nextRegion == numRegions; }
  /// Advance the walk stage.
  void advance() { nextRegion++; }
  /// Returns the next region that will be visited.
  int getNextRegion() const { return nextRegion; }

private:
  const int numRegions;
  int nextRegion;
};

namespace detail {
/// Helper templates to deduce the first argument of a callback parameter.
template <typename Ret, typename Arg, typename... Rest>
Arg first_argument_type(Ret (*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_type(Ret (F::*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_type(Ret (F::*)(Arg, Rest...) const);
template <typename F>
decltype(first_argument_type(&F::operator())) first_argument_type(F);

/// Type definition of the first argument to the given callable 'T'.
template <typename T>
using first_argument = decltype(first_argument_type(std::declval<T>()));

/// Walk all of the regions, blocks, or operations nested under (and including)
/// the given operation. The order in which regions, blocks and operations at
/// the same nesting level are visited (e.g., lexicographical or reverse
/// lexicographical order) is determined by 'Iterator'. The walk order for
/// enclosing regions, blocks and operations with respect to their nested ones
/// is specified by 'order'. These methods are invoked for void-returning
/// callbacks. A callback on a block or operation is allowed to erase that block
/// or operation only if the walk is in post-order. See non-void method for
/// pre-order erasure.
template <typename Iterator>
void walk(Operation *op, function_ref<void(Region *)> callback,
          WalkOrder order) {
  // We don't use early increment for regions because they can't be erased from
  // a callback.
  for (auto &region : Iterator::makeIterable(*op)) {
    if (order == WalkOrder::PreOrder)
      callback(&region);
    for (auto &block : Iterator::makeIterable(region)) {
      for (auto &nestedOp : Iterator::makeIterable(block))
        walk<Iterator>(&nestedOp, callback, order);
    }
    if (order == WalkOrder::PostOrder)
      callback(&region);
  }
}

template <typename Iterator>
void walk(Operation *op, function_ref<void(Block *)> callback,
          WalkOrder order) {
  for (auto &region : Iterator::makeIterable(*op)) {
    // Early increment here in the case where the block is erased.
    for (auto &block :
         llvm::make_early_inc_range(Iterator::makeIterable(region))) {
      if (order == WalkOrder::PreOrder)
        callback(&block);
      for (auto &nestedOp : Iterator::makeIterable(block))
        walk<Iterator>(&nestedOp, callback, order);
      if (order == WalkOrder::PostOrder)
        callback(&block);
    }
  }
}

template <typename Iterator>
void walk(Operation *op, function_ref<void(Operation *)> callback,
          WalkOrder order) {
  if (order == WalkOrder::PreOrder)
    callback(op);

  // TODO: This walk should be iterative over the operations.
  for (auto &region : Iterator::makeIterable(*op)) {
    for (auto &block : Iterator::makeIterable(region)) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp :
           llvm::make_early_inc_range(Iterator::makeIterable(block)))
        walk<Iterator>(&nestedOp, callback, order);
    }
  }

  if (order == WalkOrder::PostOrder)
    callback(op);
}

/// Walk all of the regions, blocks, or operations nested under (and including)
/// the given operation. The order in which regions, blocks and operations at
/// the same nesting level are visited (e.g., lexicographical or reverse
/// lexicographical order) is determined by 'Iterator'. The walk order for
/// enclosing regions, blocks and operations with respect to their nested ones
/// is specified by 'order'. This method is invoked for skippable or
/// interruptible callbacks. A callback on a block or operation is allowed to
/// erase that block or operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
template <typename Iterator>
WalkResult walk(Operation *op, function_ref<WalkResult(Region *)> callback,
                WalkOrder order) {
  // We don't use early increment for regions because they can't be erased from
  // a callback.
  for (auto &region : Iterator::makeIterable(*op)) {
    if (order == WalkOrder::PreOrder) {
      WalkResult result = callback(&region);
      if (result.wasSkipped())
        continue;
      if (result.wasInterrupted())
        return WalkResult::interrupt();
    }
    for (auto &block : Iterator::makeIterable(region)) {
      for (auto &nestedOp : Iterator::makeIterable(block))
        if (walk<Iterator>(&nestedOp, callback, order).wasInterrupted())
          return WalkResult::interrupt();
    }
    if (order == WalkOrder::PostOrder) {
      if (callback(&region).wasInterrupted())
        return WalkResult::interrupt();
      // We don't check if this region was skipped because its walk already
      // finished and the walk will continue with the next region.
    }
  }
  return WalkResult::advance();
}

template <typename Iterator>
WalkResult walk(Operation *op, function_ref<WalkResult(Block *)> callback,
                WalkOrder order) {
  for (auto &region : Iterator::makeIterable(*op)) {
    // Early increment here in the case where the block is erased.
    for (auto &block :
         llvm::make_early_inc_range(Iterator::makeIterable(region))) {
      if (order == WalkOrder::PreOrder) {
        WalkResult result = callback(&block);
        if (result.wasSkipped())
          continue;
        if (result.wasInterrupted())
          return WalkResult::interrupt();
      }
      for (auto &nestedOp : Iterator::makeIterable(block))
        if (walk<Iterator>(&nestedOp, callback, order).wasInterrupted())
          return WalkResult::interrupt();
      if (order == WalkOrder::PostOrder) {
        if (callback(&block).wasInterrupted())
          return WalkResult::interrupt();
        // We don't check if this block was skipped because its walk already
        // finished and the walk will continue with the next block.
      }
    }
  }
  return WalkResult::advance();
}

template <typename Iterator>
WalkResult walk(Operation *op, function_ref<WalkResult(Operation *)> callback,
                WalkOrder order) {
  if (order == WalkOrder::PreOrder) {
    WalkResult result = callback(op);
    // If skipped, caller will continue the walk on the next operation.
    if (result.wasSkipped())
      return WalkResult::advance();
    if (result.wasInterrupted())
      return WalkResult::interrupt();
  }

  // TODO: This walk should be iterative over the operations.
  for (auto &region : Iterator::makeIterable(*op)) {
    for (auto &block : Iterator::makeIterable(region)) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp :
           llvm::make_early_inc_range(Iterator::makeIterable(block))) {
        if (walk<Iterator>(&nestedOp, callback, order).wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }

  if (order == WalkOrder::PostOrder)
    return callback(op);
  return WalkResult::advance();
}

// Below are a set of functions to walk nested operations. Users should favor
// the direct `walk` methods on the IR classes(Operation/Block/etc) over these
// methods. They are also templated to allow for statically dispatching based
// upon the type of the callback function.

/// Walk all of the regions, blocks, or operations nested under (and including)
/// the given operation. The order in which regions, blocks and operations at
/// the same nesting level are visited (e.g., lexicographical or reverse
/// lexicographical order) is determined by 'Iterator'. The walk order for
/// enclosing regions, blocks and operations with respect to their nested ones
/// is specified by 'Order' (post-order by default). A callback on a block or
/// operation is allowed to erase that block or operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
/// This method is selected for callbacks that operate on Region*, Block*, and
/// Operation*.
///
/// Example:
///   op->walk([](Region *r) { ... });
///   op->walk([](Block *b) { ... });
///   op->walk([](Operation *op) { ... });
template <
    WalkOrder Order = WalkOrder::PostOrder, typename Iterator = ForwardIterator,
    typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
std::enable_if_t<llvm::is_one_of<ArgT, Operation *, Region *, Block *>::value,
                 RetT>
walk(Operation *op, FuncTy &&callback) {
  return detail::walk<Iterator>(op, function_ref<RetT(ArgT)>(callback), Order);
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. The order in which regions, blocks and operations at
/// the same nesting are visited (e.g., lexicographical or reverse
/// lexicographical order) is determined by 'Iterator'. The walk order for
/// enclosing regions, blocks and operations with respect to their nested ones
/// is specified by 'order' (post-order by default). This method is selected for
/// void-returning callbacks that operate on a specific derived operation type.
/// A callback on an operation is allowed to erase that operation only if the
/// walk is in post-order. See non-void method for pre-order erasure.
///
/// Example:
///   op->walk([](ReturnOp op) { ... });
template <
    WalkOrder Order = WalkOrder::PostOrder, typename Iterator = ForwardIterator,
    typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
std::enable_if_t<
    !llvm::is_one_of<ArgT, Operation *, Region *, Block *>::value &&
        std::is_same<RetT, void>::value,
    RetT>
walk(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      callback(derivedOp);
  };
  return detail::walk<Iterator>(op, function_ref<RetT(Operation *)>(wrapperFn),
                                Order);
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. The order in which regions, blocks and operations at
/// the same nesting are visited (e.g., lexicographical or reverse
/// lexicographical order) is determined by 'Iterator'. The walk order for
/// enclosing regions, blocks and operations with respect to their nested ones
/// is specified by 'Order' (post-order by default). This method is selected for
/// WalkReturn returning skippable or interruptible callbacks that operate on a
/// specific derived operation type. A callback on an operation is allowed to
/// erase that operation if either:
///   * the walk is in post-order, or
///   * the walk is in pre-order and the walk is skipped after the erasure.
///
/// Example:
///   op->walk([](ReturnOp op) {
///     if (some_invariant)
///       return WalkResult::skip();
///     if (another_invariant)
///       return WalkResult::interrupt();
///     return WalkResult::advance();
///   });
template <
    WalkOrder Order = WalkOrder::PostOrder, typename Iterator = ForwardIterator,
    typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
std::enable_if_t<
    !llvm::is_one_of<ArgT, Operation *, Region *, Block *>::value &&
        std::is_same<RetT, WalkResult>::value,
    RetT>
walk(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      return callback(derivedOp);
    return WalkResult::advance();
  };
  return detail::walk<Iterator>(op, function_ref<RetT(Operation *)>(wrapperFn),
                                Order);
}

/// Generic walkers with stage aware callbacks.

/// Walk all the operations nested under (and including) the given operation,
/// with the callback being invoked on each operation N+1 times, where N is the
/// number of regions attached to the operation. The `stage` input to the
/// callback indicates the current walk stage. This method is invoked for void
/// returning callbacks.
void walk(Operation *op,
          function_ref<void(Operation *, const WalkStage &stage)> callback);

/// Walk all the operations nested under (and including) the given operation,
/// with the callback being invoked on each operation N+1 times, where N is the
/// number of regions attached to the operation. The `stage` input to the
/// callback indicates the current walk stage. This method is invoked for
/// skippable or interruptible callbacks.
WalkResult
walk(Operation *op,
     function_ref<WalkResult(Operation *, const WalkStage &stage)> callback);

/// Walk all of the operations nested under and including the given operation.
/// This method is selected for stage-aware callbacks that operate on
/// Operation*.
///
/// Example:
///   op->walk([](Operation *op, const WalkStage &stage) { ... });
template <typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
          typename RetT = decltype(std::declval<FuncTy>()(
              std::declval<ArgT>(), std::declval<const WalkStage &>()))>
std::enable_if_t<std::is_same<ArgT, Operation *>::value, RetT>
walk(Operation *op, FuncTy &&callback) {
  return detail::walk(op,
                      function_ref<RetT(ArgT, const WalkStage &)>(callback));
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. This method is selected for void returning callbacks that
/// operate on a specific derived operation type.
///
/// Example:
///   op->walk([](ReturnOp op, const WalkStage &stage) { ... });
template <typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
          typename RetT = decltype(std::declval<FuncTy>()(
              std::declval<ArgT>(), std::declval<const WalkStage &>()))>
std::enable_if_t<!std::is_same<ArgT, Operation *>::value &&
                     std::is_same<RetT, void>::value,
                 RetT>
walk(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op, const WalkStage &stage) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      callback(derivedOp, stage);
  };
  return detail::walk(
      op, function_ref<RetT(Operation *, const WalkStage &)>(wrapperFn));
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. This method is selected for WalkReturn returning
/// interruptible callbacks that operate on a specific derived operation type.
///
/// Example:
///   op->walk(op, [](ReturnOp op, const WalkStage &stage) {
///     if (some_invariant)
///       return WalkResult::interrupt();
///     return WalkResult::advance();
///   });
template <typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
          typename RetT = decltype(std::declval<FuncTy>()(
              std::declval<ArgT>(), std::declval<const WalkStage &>()))>
std::enable_if_t<!std::is_same<ArgT, Operation *>::value &&
                     std::is_same<RetT, WalkResult>::value,
                 RetT>
walk(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op, const WalkStage &stage) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      return callback(derivedOp, stage);
    return WalkResult::advance();
  };
  return detail::walk(
      op, function_ref<RetT(Operation *, const WalkStage &)>(wrapperFn));
}

/// Utility to provide the return type of a templated walk method.
template <typename FnT>
using walkResultType = decltype(walk(nullptr, std::declval<FnT>()));
} // namespace detail

} // namespace mlir

#endif
