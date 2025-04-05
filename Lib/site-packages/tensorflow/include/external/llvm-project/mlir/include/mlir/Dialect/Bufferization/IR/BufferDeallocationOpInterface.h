//===- BufferDeallocationOpInterface.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERDEALLOCATIONOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERDEALLOCATIONOPINTERFACE_H_

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace bufferization {

/// Compare two SSA values in a deterministic manner. Two block arguments are
/// ordered by argument number, block arguments are always less than operation
/// results, and operation results are ordered by the `isBeforeInBlock` order of
/// their defining operation.
struct ValueComparator {
  bool operator()(const Value &lhs, const Value &rhs) const;
};

/// This class is used to track the ownership of values. The ownership can
/// either be not initialized yet ('Uninitialized' state), set to a unique SSA
/// value which indicates the ownership at runtime (or statically if it is a
/// constant value) ('Unique' state), or it cannot be represented in a single
/// SSA value ('Unknown' state). An artificial example of a case where ownership
/// cannot be represented in a single i1 SSA value could be the following:
/// `%0 = test.non_deterministic_select %arg0, %arg1 : i32`
/// Since the operation does not provide us a separate boolean indicator on
/// which of the two operands was selected, we would need to either insert an
/// alias check at runtime to determine if `%0` aliases with `%arg0` or `%arg1`,
/// or insert a `bufferization.clone` operation to get a fresh buffer which we
/// could assign ownership to.
///
/// The three states this class can represent form a lattice on a partial order:
/// forall X in SSA values. uninitialized < unique(X) < unknown
/// forall X, Y in SSA values.
///   unique(X) == unique(Y) iff X and Y always evaluate to the same value
///   unique(X) != unique(Y) otherwise
class Ownership {
public:
  /// Constructor that creates an 'Uninitialized' ownership. This is needed for
  /// default-construction when used in DenseMap.
  Ownership() = default;

  /// Constructor that creates an 'Unique' ownership. This is a non-explicit
  /// constructor to allow implicit conversion from 'Value'.
  Ownership(Value indicator);

  /// Get an ownership value in 'Unknown' state.
  static Ownership getUnknown();
  /// Get an ownership value in 'Unique' state with 'indicator' as parameter.
  static Ownership getUnique(Value indicator);
  /// Get an ownership value in 'Uninitialized' state.
  static Ownership getUninitialized();

  /// Check if this ownership value is in the 'Uninitialized' state.
  bool isUninitialized() const;
  /// Check if this ownership value is in the 'Unique' state.
  bool isUnique() const;
  /// Check if this ownership value is in the 'Unknown' state.
  bool isUnknown() const;

  /// If this ownership value is in 'Unique' state, this function can be used to
  /// get the indicator parameter. Using this function in any other state is UB.
  Value getIndicator() const;

  /// Get the join of the two-element subset {this,other}. Does not modify
  /// 'this'.
  Ownership getCombined(Ownership other) const;

  /// Modify 'this' ownership to be the join of the current 'this' and 'other'.
  void combine(Ownership other);

private:
  enum class State {
    Uninitialized,
    Unique,
    Unknown,
  };

  // The indicator value is only relevant in the 'Unique' state.
  Value indicator;
  State state = State::Uninitialized;
};

/// Options for BufferDeallocationOpInterface-based buffer deallocation.
struct DeallocationOptions {
  // A pass option indicating whether private functions should be modified to
  // pass the ownership of MemRef values instead of adhering to the function
  // boundary ABI.
  bool privateFuncDynamicOwnership = false;
};

/// This class collects all the state that we need to perform the buffer
/// deallocation pass with associated helper functions such that we have easy
/// access to it in the BufferDeallocationOpInterface implementations and the
/// BufferDeallocation pass.
class DeallocationState {
public:
  DeallocationState(Operation *op);

  // The state should always be passed by reference.
  DeallocationState(const DeallocationState &) = delete;

  /// Small helper function to update the ownership map by taking the current
  /// ownership ('Uninitialized' state if not yet present), computing the join
  /// with the passed ownership and storing this new value in the map. By
  /// default, it will be performed for the block where 'owned' is defined. If
  /// the ownership of the given value should be updated for another block, the
  /// 'block' argument can be explicitly passed.
  void updateOwnership(Value memref, Ownership ownership,
                       Block *block = nullptr);

  /// Removes ownerships associated with all values in the passed range for
  /// 'block'.
  void resetOwnerships(ValueRange memrefs, Block *block);

  /// Returns the ownership of 'memref' for the given basic block.
  Ownership getOwnership(Value memref, Block *block) const;

  /// Remember the given 'memref' to deallocate it at the end of the 'block'.
  void addMemrefToDeallocate(Value memref, Block *block);

  /// Forget about a MemRef that we originally wanted to deallocate at the end
  /// of 'block', possibly because it already gets deallocated before the end of
  /// the block.
  void dropMemrefToDeallocate(Value memref, Block *block);

  /// Return a sorted list of MemRef values which are live at the start of the
  /// given block.
  void getLiveMemrefsIn(Block *block, SmallVectorImpl<Value> &memrefs);

  /// Given an SSA value of MemRef type, this function queries the ownership and
  /// if it is not already in the 'Unique' state, potentially inserts IR to get
  /// a new SSA value, returned as the first element of the pair, which has
  /// 'Unique' ownership and can be used instead of the passed Value with the
  /// the ownership indicator returned as the second element of the pair.
  std::pair<Value, Value>
  getMemrefWithUniqueOwnership(OpBuilder &builder, Value memref, Block *block);

  /// Given two basic blocks and the values passed via block arguments to the
  /// destination block, compute the list of MemRefs that have to be retained in
  /// the 'fromBlock' to not run into a use-after-free situation.
  /// This list consists of the MemRefs in the successor operand list of the
  /// terminator and the MemRefs in the 'out' set of the liveness analysis
  /// intersected with the 'in' set of the destination block.
  ///
  /// toRetain = filter(successorOperands + (liveOut(fromBlock) insersect
  ///   liveIn(toBlock)), isMemRef)
  void getMemrefsToRetain(Block *fromBlock, Block *toBlock,
                          ValueRange destOperands,
                          SmallVectorImpl<Value> &toRetain) const;

  /// For a given block, computes the list of MemRefs that potentially need to
  /// be deallocated at the end of that block. This list also contains values
  /// that have to be retained (and are thus part of the list returned by
  /// `getMemrefsToRetain`) and is computed by taking the MemRefs in the 'in'
  /// set of the liveness analysis of 'block'  appended by the set of MemRefs
  /// allocated in 'block' itself and subtracted by the set of MemRefs
  /// deallocated in 'block'.
  /// Note that we don't have to take the intersection of the liveness 'in' set
  /// with the 'out' set of the predecessor block because a value that is in the
  /// 'in' set must be defined in an ancestor block that dominates all direct
  /// predecessors and thus the 'in' set of this block is a subset of the 'out'
  /// sets of each predecessor.
  ///
  /// memrefs = filter((liveIn(block) U
  ///   allocated(block) U arguments(block)) \ deallocated(block), isMemRef)
  ///
  /// The list of conditions is then populated by querying the internal
  /// datastructures for the ownership value of that MemRef.
  LogicalResult
  getMemrefsAndConditionsToDeallocate(OpBuilder &builder, Location loc,
                                      Block *block,
                                      SmallVectorImpl<Value> &memrefs,
                                      SmallVectorImpl<Value> &conditions) const;

  /// Returns the symbol cache to lookup functions from call operations to check
  /// attributes on the function operation.
  SymbolTableCollection *getSymbolTable() { return &symbolTable; }

private:
  // Symbol cache to lookup functions from call operations to check attributes
  // on the function operation.
  SymbolTableCollection symbolTable;

  // Mapping from each SSA value with MemRef type to the associated ownership in
  // each block.
  DenseMap<std::pair<Value, Block *>, Ownership> ownershipMap;

  // Collects the list of MemRef values that potentially need to be deallocated
  // per block. It is also fine (albeit not efficient) to add MemRef values that
  // don't have to be deallocated, but only when the ownership is not 'Unknown'.
  DenseMap<Block *, SmallVector<Value>> memrefsToDeallocatePerBlock;

  // The underlying liveness analysis to compute fine grained information about
  // alloc and dealloc positions.
  Liveness liveness;
};

namespace deallocation_impl {
/// Insert a `bufferization.dealloc` operation right before `op` which has to be
/// a terminator without any successors. Note that it is not required to have
/// the ReturnLike trait attached. The MemRef values in the `operands` argument
/// will be added to the list of retained values and their updated ownership
/// values will be appended to the `updatedOperandOwnerships` list. `op` is not
/// modified in any way. Returns failure if at least one of the MemRefs to
/// deallocate does not have 'Unique' ownership (likely as a result of an
/// incorrect implementation of the `process` or
/// `materializeUniqueOwnershipForMemref` interface method) or the original
/// `op`.
FailureOr<Operation *>
insertDeallocOpForReturnLike(DeallocationState &state, Operation *op,
                             ValueRange operands,
                             SmallVectorImpl<Value> &updatedOperandOwnerships);
} // namespace deallocation_impl

} // namespace bufferization
} // namespace mlir

//===----------------------------------------------------------------------===//
// Buffer Deallocation Interface
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERDEALLOCATIONOPINTERFACE_H_
