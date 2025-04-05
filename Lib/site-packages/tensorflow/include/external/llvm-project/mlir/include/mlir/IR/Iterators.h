//===- Iterators.h - IR iterators for IR visitors ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The iterators defined in this file can be used together with IR visitors.
// Note: These iterators cannot be defined in Visitors.h because that would
// introduce a cyclic header dependency due to Operation.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ITERATORS_H
#define MLIR_IR_ITERATORS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

namespace mlir {
/// This iterator enumerates elements in "reverse" order. It is a wrapper around
/// llvm::reverse.
struct ReverseIterator {
  // llvm::reverse uses RangeT::rbegin and RangeT::rend.
  template <typename RangeT>
  static constexpr auto makeIterable(RangeT &&range) {
    return llvm::reverse(
        ForwardIterator::makeIterable(std::forward<RangeT>(range)));
  }
};

/// This iterator enumerates elements according to their dominance relationship.
/// Operations and regions are enumerated in "forward" order. Blocks are
/// enumerated according to their successor relationship. Unreachable blocks are
/// not enumerated. Blocks may not be erased during the traversal.
///
/// Note: If `NoGraphRegions` is set to "true", this iterator asserts that each
/// visited region has SSA dominance. In either case, the ops in such regions
/// are visited in forward order, but for regions without SSA dominance this
/// does not guarantee that defining ops are visited before their users.
template <bool NoGraphRegions = false>
struct ForwardDominanceIterator {
  static Block &makeIterable(Block &range) {
    return ForwardIterator::makeIterable(range);
  }

  static auto makeIterable(Region &region) {
    if (NoGraphRegions) {
      // Only regions with SSA dominance are allowed.
      assert(mayHaveSSADominance(region) && "graph regions are not allowed");
    }

    // Create DFS iterator. Blocks are enumerated according to their successor
    // relationship.
    Block *null = nullptr;
    auto it = region.empty()
                  ? llvm::make_range(llvm::df_end(null), llvm::df_end(null))
                  : llvm::depth_first(&region.front());

    // Walk API expects Block references instead of pointers.
    return llvm::make_pointee_range(it);
  }

  static MutableArrayRef<Region> makeIterable(Operation &range) {
    return ForwardIterator::makeIterable(range);
  }
};

/// This iterator enumerates elements according to their reverse dominance
/// relationship. Operations and regions are enumerated in "reverse" order.
/// Blocks are enumerated according to their successor relationship, but
/// post-order. I.e., a block is visited after its successors have been visited.
/// Cycles in the block graph are broken in an unspecified way. Unreachable
/// blocks are not enumerated. Blocks may not be erased during the traversal.
///
/// Note: If `NoGraphRegions` is set to "true", this iterator asserts that each
/// visited region has SSA dominance.
template <bool NoGraphRegions = false>
struct ReverseDominanceIterator {
  // llvm::reverse uses RangeT::rbegin and RangeT::rend.
  static constexpr auto makeIterable(Block &range) {
    return llvm::reverse(ForwardIterator::makeIterable(range));
  }

  static constexpr auto makeIterable(Operation &range) {
    return llvm::reverse(ForwardIterator::makeIterable(range));
  }

  static auto makeIterable(Region &region) {
    if (NoGraphRegions) {
      // Only regions with SSA dominance are allowed.
      assert(mayHaveSSADominance(region) && "graph regions are not allowed");
    }

    // Create post-order iterator. Blocks are enumerated according to their
    // successor relationship.
    Block *null = nullptr;
    auto it = region.empty()
                  ? llvm::make_range(llvm::po_end(null), llvm::po_end(null))
                  : llvm::post_order(&region.front());

    // Walk API expects Block references instead of pointers.
    return llvm::make_pointee_range(it);
  }
};
} // namespace mlir

#endif // MLIR_IR_ITERATORS_H
