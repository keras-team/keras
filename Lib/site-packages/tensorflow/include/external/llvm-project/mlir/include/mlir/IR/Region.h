//===- Region.h - MLIR Region Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Region class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGION_H
#define MLIR_IR_REGION_H

#include "mlir/IR/Block.h"

namespace mlir {
class TypeRange;
template <typename ValueRangeT>
class ValueTypeRange;
class IRMapping;

/// This class contains a list of basic blocks and a link to the parent
/// operation it is attached to.
class Region {
public:
  Region() = default;
  explicit Region(Operation *container);
  ~Region();

  /// Return the context this region is inserted in.  The region must have a
  /// valid parent container.
  MLIRContext *getContext();

  /// Return a location for this region. This is the location attached to the
  /// parent container. The region must have a valid parent container.
  Location getLoc();

  //===--------------------------------------------------------------------===//
  // Block list management
  //===--------------------------------------------------------------------===//

  using BlockListType = llvm::iplist<Block>;
  BlockListType &getBlocks() { return blocks; }
  Block &emplaceBlock() {
    push_back(new Block);
    return back();
  }

  // Iteration over the blocks in the region.
  using iterator = BlockListType::iterator;
  using reverse_iterator = BlockListType::reverse_iterator;

  iterator begin() { return blocks.begin(); }
  iterator end() { return blocks.end(); }
  reverse_iterator rbegin() { return blocks.rbegin(); }
  reverse_iterator rend() { return blocks.rend(); }

  bool empty() { return blocks.empty(); }
  void push_back(Block *block) { blocks.push_back(block); }
  void push_front(Block *block) { blocks.push_front(block); }

  Block &back() { return blocks.back(); }
  Block &front() { return blocks.front(); }

  /// Return true if this region has exactly one block.
  bool hasOneBlock() { return !empty() && std::next(begin()) == end(); }

  /// getSublistAccess() - Returns pointer to member of region.
  static BlockListType Region::*getSublistAccess(Block *) {
    return &Region::blocks;
  }

  //===--------------------------------------------------------------------===//
  // Argument Handling
  //===--------------------------------------------------------------------===//

  // This is the list of arguments to the block.
  using BlockArgListType = MutableArrayRef<BlockArgument>;
  BlockArgListType getArguments() {
    return empty() ? BlockArgListType() : front().getArguments();
  }

  /// Returns the argument types of the first block within the region.
  ValueTypeRange<BlockArgListType> getArgumentTypes();

  using args_iterator = BlockArgListType::iterator;
  using reverse_args_iterator = BlockArgListType::reverse_iterator;
  args_iterator args_begin() { return getArguments().begin(); }
  args_iterator args_end() { return getArguments().end(); }
  reverse_args_iterator args_rbegin() { return getArguments().rbegin(); }
  reverse_args_iterator args_rend() { return getArguments().rend(); }

  bool args_empty() { return getArguments().empty(); }

  /// Add one value to the argument list.
  BlockArgument addArgument(Type type, Location loc) {
    return front().addArgument(type, loc);
  }

  /// Insert one value to the position in the argument list indicated by the
  /// given iterator. The existing arguments are shifted. The block is expected
  /// not to have predecessors.
  BlockArgument insertArgument(args_iterator it, Type type, Location loc) {
    return front().insertArgument(it, type, loc);
  }

  /// Add one argument to the argument list for each type specified in the list.
  /// `locs` contains the locations for each of the new arguments, and must be
  /// of equal size to `types`.
  iterator_range<args_iterator> addArguments(TypeRange types,
                                             ArrayRef<Location> locs);

  /// Add one value to the argument list at the specified position.
  BlockArgument insertArgument(unsigned index, Type type, Location loc) {
    return front().insertArgument(index, type, loc);
  }

  /// Erase the argument at 'index' and remove it from the argument list.
  void eraseArgument(unsigned index) { front().eraseArgument(index); }

  unsigned getNumArguments() { return getArguments().size(); }
  BlockArgument getArgument(unsigned i) { return getArguments()[i]; }

  //===--------------------------------------------------------------------===//
  // Operation list utilities
  //===--------------------------------------------------------------------===//

  /// This class provides iteration over the held operations of blocks directly
  /// within a region.
  class OpIterator final
      : public llvm::iterator_facade_base<OpIterator, std::forward_iterator_tag,
                                          Operation> {
  public:
    /// Initialize OpIterator for a region, specify `end` to return the iterator
    /// to last operation.
    explicit OpIterator(Region *region, bool end = false);

    using llvm::iterator_facade_base<OpIterator, std::forward_iterator_tag,
                                     Operation>::operator++;
    OpIterator &operator++();
    Operation *operator->() const { return &*operation; }
    Operation &operator*() const { return *operation; }

    /// Compare this iterator with another.
    bool operator==(const OpIterator &rhs) const {
      return operation == rhs.operation;
    }
    bool operator!=(const OpIterator &rhs) const { return !(*this == rhs); }

  private:
    void skipOverBlocksWithNoOps();

    /// The region whose operations are being iterated over.
    Region *region;
    /// The block of 'region' whose operations are being iterated over.
    Region::iterator block;
    /// The current operation within 'block'.
    Block::iterator operation;
  };

  /// This class provides iteration over the held operations of a region for a
  /// specific operation type.
  template <typename OpT>
  using op_iterator = detail::op_iterator<OpT, OpIterator>;

  /// Return iterators that walk the operations nested directly within this
  /// region.
  OpIterator op_begin() { return OpIterator(this); }
  OpIterator op_end() { return OpIterator(this, /*end=*/true); }
  iterator_range<OpIterator> getOps() { return {op_begin(), op_end()}; }

  /// Return iterators that walk operations of type 'T' nested directly within
  /// this region.
  template <typename OpT>
  op_iterator<OpT> op_begin() {
    return detail::op_filter_iterator<OpT, OpIterator>(op_begin(), op_end());
  }
  template <typename OpT>
  op_iterator<OpT> op_end() {
    return detail::op_filter_iterator<OpT, OpIterator>(op_end(), op_end());
  }
  template <typename OpT>
  iterator_range<op_iterator<OpT>> getOps() {
    auto endIt = op_end();
    return {detail::op_filter_iterator<OpT, OpIterator>(op_begin(), endIt),
            detail::op_filter_iterator<OpT, OpIterator>(endIt, endIt)};
  }

  //===--------------------------------------------------------------------===//
  // Misc. utilities
  //===--------------------------------------------------------------------===//

  /// Return the region containing this region or nullptr if the region is
  /// attached to a top-level operation.
  Region *getParentRegion();

  /// Return the parent operation this region is attached to.
  Operation *getParentOp() { return container; }

  /// Find the first parent operation of the given type, or nullptr if there is
  /// no ancestor operation.
  template <typename ParentT>
  ParentT getParentOfType() {
    auto *region = this;
    do {
      if (auto parent = dyn_cast_or_null<ParentT>(region->container))
        return parent;
    } while ((region = region->getParentRegion()));
    return ParentT();
  }

  /// Return the number of this region in the parent operation.
  unsigned getRegionNumber();

  /// Return true if this region is a proper ancestor of the `other` region.
  bool isProperAncestor(Region *other);

  /// Return true if this region is ancestor of the `other` region.  A region
  /// is considered as its own ancestor, use `isProperAncestor` to avoid this.
  bool isAncestor(Region *other) {
    return this == other || isProperAncestor(other);
  }

  /// Clone the internal blocks from this region into dest. Any
  /// cloned blocks are appended to the back of dest. If the mapper
  /// contains entries for block arguments, these arguments are not included
  /// in the respective cloned block.
  ///
  /// Calling this method from multiple threads is generally safe if through the
  /// process of cloning, no new uses of 'Value's from outside the region are
  /// created. Using the mapper, it is possible to avoid adding uses to outside
  /// operands by remapping them to 'Value's owned by the caller thread.
  void cloneInto(Region *dest, IRMapping &mapper);
  /// Clone this region into 'dest' before the given position in 'dest'.
  void cloneInto(Region *dest, Region::iterator destPos, IRMapping &mapper);

  /// Takes body of another region (that region will have no body after this
  /// operation completes).  The current body of this region is cleared.
  void takeBody(Region &other) {
    dropAllReferences();
    blocks.clear();
    blocks.splice(blocks.end(), other.getBlocks());
  }

  /// Returns 'block' if 'block' lies in this region, or otherwise finds the
  /// ancestor of 'block' that lies in this region. Returns nullptr if the
  /// latter fails.
  Block *findAncestorBlockInRegion(Block &block);

  /// Returns 'op' if 'op' lies in this region, or otherwise finds the
  /// ancestor of 'op' that lies in this region. Returns nullptr if the
  /// latter fails.
  Operation *findAncestorOpInRegion(Operation &op);

  /// Drop all operand uses from operations within this region, which is
  /// an essential step in breaking cyclic dependences between references when
  /// they are to be deleted.
  void dropAllReferences();

  //===--------------------------------------------------------------------===//
  // Walkers
  //===--------------------------------------------------------------------===//

  /// Walk all nested operations, blocks or regions (including this region),
  /// depending on the type of callback.
  ///
  /// The order in which operations, blocks or regions at the same nesting
  /// level are visited (e.g., lexicographical or reverse lexicographical order)
  /// is determined by `Iterator`. The walk order for enclosing operations,
  /// blocks or regions with respect to their nested ones is specified by
  /// `Order` (post-order by default).
  ///
  /// A callback on a operation or block is allowed to erase that operation or
  /// block if either:
  ///   * the walk is in post-order, or
  ///   * the walk is in pre-order and the walk is skipped after the erasure.
  ///
  /// See Operation::walk for more details.
  template <WalkOrder Order = WalkOrder::PostOrder,
            typename Iterator = ForwardIterator, typename FnT,
            typename ArgT = detail::first_argument<FnT>,
            typename RetT = detail::walkResultType<FnT>>
  RetT walk(FnT &&callback) {
    if constexpr (std::is_same<ArgT, Region *>::value &&
                  Order == WalkOrder::PreOrder) {
      // Pre-order walk on regions: invoke the callback on this region.
      if constexpr (std::is_same<RetT, void>::value) {
        callback(this);
      } else {
        RetT result = callback(this);
        if (result.wasSkipped())
          return WalkResult::advance();
        if (result.wasInterrupted())
          return WalkResult::interrupt();
      }
    }

    // Walk nested operations, blocks or regions.
    for (auto &block : *this) {
      if constexpr (std::is_same<RetT, void>::value) {
        block.walk<Order, Iterator>(callback);
      } else {
        if (block.walk<Order, Iterator>(callback).wasInterrupted())
          return WalkResult::interrupt();
      }
    }

    if constexpr (std::is_same<ArgT, Region *>::value &&
                  Order == WalkOrder::PostOrder) {
      // Post-order walk on regions: invoke the callback on this block.
      return callback(this);
    }
    if constexpr (!std::is_same<RetT, void>::value)
      return WalkResult::advance();
  }

  //===--------------------------------------------------------------------===//
  // CFG view utilities
  //===--------------------------------------------------------------------===//

  /// Displays the CFG in a window. This is for use from the debugger and
  /// depends on Graphviz to generate the graph.
  /// This function is defined in ViewOpGraph.cpp and only works with that
  /// target linked.
  void viewGraph(const Twine &regionName);
  void viewGraph();

private:
  BlockListType blocks;

  /// This is the object we are part of.
  Operation *container = nullptr;
};

/// This class provides an abstraction over the different types of ranges over
/// Regions. In many cases, this prevents the need to explicitly materialize a
/// SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class RegionRange
    : public llvm::detail::indexed_accessor_range_base<
          RegionRange,
          PointerUnion<Region *, const std::unique_ptr<Region> *, Region **>,
          Region *, Region *, Region *> {
  /// The type representing the owner of this range. This is either an owning
  /// list of regions, a list of region unique pointers, or a list of region
  /// pointers.
  using OwnerT =
      PointerUnion<Region *, const std::unique_ptr<Region> *, Region **>;

public:
  using RangeBaseT::RangeBaseT;

  RegionRange(MutableArrayRef<Region> regions = std::nullopt);

  template <typename Arg, typename = std::enable_if_t<std::is_constructible<
                              ArrayRef<std::unique_ptr<Region>>, Arg>::value>>
  RegionRange(Arg &&arg)
      : RegionRange(ArrayRef<std::unique_ptr<Region>>(std::forward<Arg>(arg))) {
  }
  template <typename Arg>
  RegionRange(
      Arg &&arg,
      std::enable_if_t<std::is_constructible<ArrayRef<Region *>, Arg>::value>
          * = nullptr)
      : RegionRange(ArrayRef<Region *>(std::forward<Arg>(arg))) {}
  RegionRange(ArrayRef<std::unique_ptr<Region>> regions);
  RegionRange(ArrayRef<Region *> regions);

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(const OwnerT &owner, ptrdiff_t index);
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Region *dereference_iterator(const OwnerT &owner, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

} // namespace mlir

#endif // MLIR_IR_REGION_H
