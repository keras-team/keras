//===- BufferUtils.h - Buffer optimization utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for passes optimizing code that has already
// been converted to buffers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERUTILS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERUTILS_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace memref {
class GlobalOp;
} // namespace memref

namespace bufferization {

/// A simple analysis that detects allocation operations.
class BufferPlacementAllocs {
public:
  /// Represents a tuple of allocValue and deallocOperation.
  using AllocEntry = std::tuple<Value, Operation *>;

  /// Represents a list containing all alloc entries.
  using AllocEntryList = SmallVector<AllocEntry, 8>;

  /// Get the start operation to place the given alloc value within the
  /// specified placement block.
  static Operation *getStartOperation(Value allocValue, Block *placementBlock,
                                      const Liveness &liveness);

public:
  /// Initializes the internal list by discovering all supported allocation
  /// nodes.
  BufferPlacementAllocs(Operation *op);

  /// Returns the begin iterator to iterate over all allocations.
  AllocEntryList::const_iterator begin() const { return allocs.begin(); }

  /// Returns the end iterator that can be used in combination with begin.
  AllocEntryList::const_iterator end() const { return allocs.end(); }

  /// Returns the begin iterator to iterate over all allocations.
  AllocEntryList::iterator begin() { return allocs.begin(); }

  /// Returns the end iterator that can be used in combination with begin.
  AllocEntryList::iterator end() { return allocs.end(); }

  /// Registers a new allocation entry.
  void registerAlloc(const AllocEntry &entry) { allocs.push_back(entry); }

private:
  /// Searches for and registers all supported allocation entries.
  void build(Operation *op);

private:
  /// Maps allocation nodes to their associated blocks.
  AllocEntryList allocs;
};

/// Finds a common dominator for the given value while taking the positions
/// of the values in the value set into account. It supports dominator and
/// post-dominator analyses via template arguments. If no common dominator
/// can be found, this function will return "nullptr".
template <typename DominatorT>
Block *findCommonDominator(Value value,
                           const BufferViewFlowAnalysis::ValueSetT &values,
                           const DominatorT &doms) {
  // Store blocks in a set before querying `DominanceInfo` to filter out
  // duplicate blocks (for performance reasons).
  llvm::SmallPtrSet<Block *, 16> blocks;
  // Start with the current block the value is defined in.
  blocks.insert(value.getParentBlock());
  for (Value childValue : values) {
    for (Operation *user : childValue.getUsers()) {
      // Find an appropriate dominator block that takes the current use into
      // account.
      blocks.insert(user->getBlock());
    }
    // Take values without any users into account.
    blocks.insert(childValue.getParentBlock());
  }
  return doms.findNearestCommonDominator(blocks);
}

/// The base class for all BufferPlacement transformations.
class BufferPlacementTransformationBase {
public:
  using ValueSetT = BufferViewFlowAnalysis::ValueSetT;

  /// Constructs a new operation base using the given root operation.
  BufferPlacementTransformationBase(Operation *op);

protected:
  /// Alias information that can be updated during the insertion of copies.
  BufferViewFlowAnalysis aliases;

  /// Stores all internally managed allocations.
  BufferPlacementAllocs allocs;

  /// The underlying liveness analysis to compute fine grained information
  /// about alloc and dealloc positions.
  Liveness liveness;
};

// Create a global op for the given tensor-valued constant in the program.
// Globals are created lazily at the top of the enclosing ModuleOp with pretty
// names. Duplicates are avoided.
FailureOr<memref::GlobalOp> getGlobalFor(arith::ConstantOp constantOp,
                                         uint64_t alignment,
                                         Attribute memorySpace = {});

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERUTILS_H
