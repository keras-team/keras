//===- BlockSupport.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of support types for the Block class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BLOCKSUPPORT_H
#define MLIR_IR_BLOCKSUPPORT_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {
class Block;

//===----------------------------------------------------------------------===//
// BlockOperand
//===----------------------------------------------------------------------===//

/// A block operand represents an operand that holds a reference to a Block,
/// e.g. for terminator operations.
class BlockOperand : public IROperand<BlockOperand, Block *> {
public:
  using IROperand<BlockOperand, Block *>::IROperand;

  /// Provide the use list that is attached to the given block.
  static IRObjectWithUseList<BlockOperand> *getUseList(Block *value);

  /// Return which operand this is in the BlockOperand list of the Operation.
  unsigned getOperandNumber();
};

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

/// Implement a predecessor iterator for blocks. This works by walking the use
/// lists of the blocks. The entries on this list are the BlockOperands that
/// are embedded into terminator operations. From the operand, we can get the
/// terminator that contains it, and its parent block is the predecessor.
class PredecessorIterator final
    : public llvm::mapped_iterator<ValueUseIterator<BlockOperand>,
                                   Block *(*)(BlockOperand &)> {
  static Block *unwrap(BlockOperand &value);

public:
  /// Initializes the operand type iterator to the specified operand iterator.
  PredecessorIterator(ValueUseIterator<BlockOperand> it)
      : llvm::mapped_iterator<ValueUseIterator<BlockOperand>,
                              Block *(*)(BlockOperand &)>(it, &unwrap) {}
  explicit PredecessorIterator(BlockOperand *operand)
      : PredecessorIterator(ValueUseIterator<BlockOperand>(operand)) {}

  /// Get the successor number in the predecessor terminator.
  unsigned getSuccessorIndex() const;
};

//===----------------------------------------------------------------------===//
// Successors
//===----------------------------------------------------------------------===//

/// This class implements the successor iterators for Block.
class SuccessorRange final
    : public llvm::detail::indexed_accessor_range_base<
          SuccessorRange, BlockOperand *, Block *, Block *, Block *> {
public:
  using RangeBaseT::RangeBaseT;
  SuccessorRange();
  SuccessorRange(Block *block);
  SuccessorRange(Operation *term);

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static BlockOperand *offset_base(BlockOperand *object, ptrdiff_t index) {
    return object + index;
  }
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Block *dereference_iterator(BlockOperand *object, ptrdiff_t index) {
    return object[index].get();
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// BlockRange
//===----------------------------------------------------------------------===//

/// This class provides an abstraction over the different types of ranges over
/// Blocks. In many cases, this prevents the need to explicitly materialize a
/// SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class BlockRange final
    : public llvm::detail::indexed_accessor_range_base<
          BlockRange, llvm::PointerUnion<BlockOperand *, Block *const *>,
          Block *, Block *, Block *> {
public:
  using RangeBaseT::RangeBaseT;
  BlockRange(ArrayRef<Block *> blocks = std::nullopt);
  BlockRange(SuccessorRange successors);
  template <typename Arg, typename = std::enable_if_t<std::is_constructible<
                              ArrayRef<Block *>, Arg>::value>>
  BlockRange(Arg &&arg)
      : BlockRange(ArrayRef<Block *>(std::forward<Arg>(arg))) {}
  BlockRange(std::initializer_list<Block *> blocks)
      : BlockRange(ArrayRef<Block *>(blocks)) {}

private:
  /// The owner of the range is either:
  /// * A pointer to the first element of an array of block operands.
  /// * A pointer to the first element of an array of Block *.
  using OwnerT = llvm::PointerUnion<BlockOperand *, Block *const *>;

  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(OwnerT object, ptrdiff_t index);

  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Block *dereference_iterator(OwnerT object, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// Operation Iterators
//===----------------------------------------------------------------------===//

namespace detail {
/// A utility iterator that filters out operations that are not 'OpT'.
template <typename OpT, typename IteratorT>
class op_filter_iterator
    : public llvm::filter_iterator<IteratorT, bool (*)(Operation &)> {
  static bool filter(Operation &op) { return llvm::isa<OpT>(op); }

public:
  op_filter_iterator(IteratorT it, IteratorT end)
      : llvm::filter_iterator<IteratorT, bool (*)(Operation &)>(it, end,
                                                                &filter) {}

  /// Allow implicit conversion to the underlying iterator.
  operator const IteratorT &() const { return this->wrapped(); }
};

/// This class provides iteration over the held operations of a block for a
/// specific operation type.
template <typename OpT, typename IteratorT>
class op_iterator
    : public llvm::mapped_iterator<op_filter_iterator<OpT, IteratorT>,
                                   OpT (*)(Operation &)> {
  static OpT unwrap(Operation &op) { return cast<OpT>(op); }

public:
  /// Initializes the iterator to the specified filter iterator.
  op_iterator(op_filter_iterator<OpT, IteratorT> it)
      : llvm::mapped_iterator<op_filter_iterator<OpT, IteratorT>,
                              OpT (*)(Operation &)>(it, &unwrap) {}

  /// Allow implicit conversion to the underlying block iterator.
  operator const IteratorT &() const { return this->wrapped(); }
};
} // namespace detail
} // namespace mlir

namespace llvm {

/// Provide support for hashing successor ranges.
template <>
struct DenseMapInfo<mlir::SuccessorRange> {
  static mlir::SuccessorRange getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<mlir::BlockOperand *>::getEmptyKey();
    return mlir::SuccessorRange(pointer, 0);
  }
  static mlir::SuccessorRange getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<mlir::BlockOperand *>::getTombstoneKey();
    return mlir::SuccessorRange(pointer, 0);
  }
  static unsigned getHashValue(mlir::SuccessorRange value) {
    return llvm::hash_combine_range(value.begin(), value.end());
  }
  static bool isEqual(mlir::SuccessorRange lhs, mlir::SuccessorRange rhs) {
    if (rhs.getBase() == getEmptyKey().getBase())
      return lhs.getBase() == getEmptyKey().getBase();
    if (rhs.getBase() == getTombstoneKey().getBase())
      return lhs.getBase() == getTombstoneKey().getBase();
    return lhs == rhs;
  }
};

//===----------------------------------------------------------------------===//
// ilist_traits for Operation
//===----------------------------------------------------------------------===//

namespace ilist_detail {
// Explicitly define the node access for the operation list so that we can
// break the dependence on the Operation class in this header. This allows for
// operations to have trailing Regions without a circular include
// dependence.
template <>
struct SpecificNodeAccess<
    typename compute_node_options<::mlir::Operation>::type> : NodeAccess {
protected:
  using OptionsT = typename compute_node_options<mlir::Operation>::type;
  using pointer = typename OptionsT::pointer;
  using const_pointer = typename OptionsT::const_pointer;
  using node_type = ilist_node_impl<OptionsT>;

  static node_type *getNodePtr(pointer N);
  static const node_type *getNodePtr(const_pointer N);

  static pointer getValuePtr(node_type *N);
  static const_pointer getValuePtr(const node_type *N);
};
} // namespace ilist_detail

template <>
struct ilist_traits<::mlir::Operation> {
  using Operation = ::mlir::Operation;
  using op_iterator = simple_ilist<Operation>::iterator;

  static void deleteNode(Operation *op);
  void addNodeToList(Operation *op);
  void removeNodeFromList(Operation *op);
  void transferNodesFromList(ilist_traits<Operation> &otherList,
                             op_iterator first, op_iterator last);

private:
  mlir::Block *getContainingBlock();
};

//===----------------------------------------------------------------------===//
// ilist_traits for Block
//===----------------------------------------------------------------------===//

template <>
struct ilist_traits<::mlir::Block> : public ilist_alloc_traits<::mlir::Block> {
  using Block = ::mlir::Block;
  using block_iterator = simple_ilist<::mlir::Block>::iterator;

  void addNodeToList(Block *block);
  void removeNodeFromList(Block *block);
  void transferNodesFromList(ilist_traits<Block> &otherList,
                             block_iterator first, block_iterator last);

private:
  mlir::Region *getParentRegion();
};

} // namespace llvm

#endif // MLIR_IR_BLOCKSUPPORT_H
