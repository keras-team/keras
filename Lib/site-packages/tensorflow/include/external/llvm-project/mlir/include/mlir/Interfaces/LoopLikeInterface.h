//===- LoopLikeInterface.h - Loop-like operations interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for loop like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
#define MLIR_INTERFACES_LOOPLIKEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class RewriterBase;

/// A function that returns the additional yielded values during
/// `replaceWithAdditionalYields`. `newBbArgs` are the newly added region
/// iter_args. This function should return as many values as there are block
/// arguments in `newBbArgs`.
using NewYieldValuesFn = std::function<SmallVector<Value>(
    OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs)>;

namespace detail {
/// Verify invariants of the LoopLikeOpInterface.
LogicalResult verifyLoopLikeOpInterface(Operation *op);
} // namespace detail

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

namespace OpTrait {
// A trait indicating that the single region contained in the operation has
// parallel execution semantics. This may have implications in a certain pass.
// For example, buffer hoisting is illegal in parallel loops, and local buffers
// may be accessed by parallel threads simultaneously.
template <typename ConcreteType>
class HasParallelRegion : public TraitBase<ConcreteType, HasParallelRegion> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyOneRegion(op);
  }
};

} // namespace OpTrait

// Gathers all maximal sub-blocks of operations that do not themselves
// include a `OpTy` (an operation could have a descendant `OpTy` though
// in its tree). Ignores the block terminators.
template <typename OpTy>
struct JamBlockGatherer {
  // Store iterators to the first and last op of each sub-block found.
  SmallVector<std::pair<Block::iterator, Block::iterator>> subBlocks;

  // This is a linear time walk.
  void walk(Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        walk(block);
  }

  void walk(Block &block) {
    assert(!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>() &&
           "expected block to have a terminator");
    for (Block::iterator it = block.begin(), e = std::prev(block.end());
         it != e;) {
      Block::iterator subBlockStart = it;
      while (it != e && !isa<OpTy>(&*it))
        ++it;
      if (it != subBlockStart)
        subBlocks.emplace_back(subBlockStart, std::prev(it));
      // Process all for ops that appear next.
      while (it != e && isa<OpTy>(&*it))
        walk(&*it++);
    }
  }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated interface declarations.
#include "mlir/Interfaces/LoopLikeInterface.h.inc"

#endif // MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
