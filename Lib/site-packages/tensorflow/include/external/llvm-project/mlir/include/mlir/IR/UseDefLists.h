//===- UseDefLists.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic use/def list machinery and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_USEDEFLISTS_H
#define MLIR_IR_USEDEFLISTS_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/iterator_range.h"

namespace mlir {

class Operation;
template <typename OperandType>
class ValueUseIterator;
template <typename UseIteratorT, typename OperandType>
class ValueUserIterator;

//===----------------------------------------------------------------------===//
// IROperand
//===----------------------------------------------------------------------===//

namespace detail {
/// This class is the base for IROperand, and provides all of the non-templated
/// facilities for operand use management.
class IROperandBase {
public:
  /// Return the owner of this operand.
  Operation *getOwner() const { return owner; }

  /// Return the next operand on the use-list of the value we are referring to.
  /// This should generally only be used by the internal implementation details
  /// of the SSA machinery.
  IROperandBase *getNextOperandUsingThisValue() { return nextUse; }

  /// Initialize the use-def chain by setting the back address to self and
  /// nextUse to nullptr.
  void initChainWithUse(IROperandBase **self) {
    assert(this == *self);
    back = self;
    nextUse = nullptr;
  }

  /// Link the current node to next.
  void linkTo(IROperandBase *next) {
    nextUse = next;
    if (nextUse)
      nextUse->back = &nextUse;
  }

protected:
  IROperandBase(Operation *owner) : owner(owner) {}
  IROperandBase(IROperandBase &&other) : owner(other.owner) {
    *this = std::move(other);
  }
  IROperandBase &operator=(IROperandBase &&other) {
    removeFromCurrent();
    other.removeFromCurrent();
    other.back = nullptr;
    nextUse = nullptr;
    back = nullptr;
    return *this;
  }
  /// Operands are not copyable or assignable.
  IROperandBase(const IROperandBase &use) = delete;
  IROperandBase &operator=(const IROperandBase &use) = delete;

  ~IROperandBase() { removeFromCurrent(); }

  /// Remove this use of the operand.
  void drop() {
    removeFromCurrent();
    nextUse = nullptr;
    back = nullptr;
  }

  /// Remove this operand from the current use list.
  void removeFromCurrent() {
    if (!back)
      return;
    *back = nextUse;
    if (nextUse)
      nextUse->back = back;
  }

  /// Insert this operand into the given use list.
  template <typename UseListT>
  void insertInto(UseListT *useList) {
    back = &useList->firstUse;
    nextUse = useList->firstUse;
    if (nextUse)
      nextUse->back = &nextUse;
    useList->firstUse = this;
  }

  /// The next operand in the use-chain.
  IROperandBase *nextUse = nullptr;

  /// This points to the previous link in the use-chain.
  IROperandBase **back = nullptr;

private:
  /// The operation owner of this operand.
  Operation *const owner;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// IROperand
//===----------------------------------------------------------------------===//

/// A reference to a value, suitable for use as an operand of an operation.
/// IRValueT is the root type to use for values this tracks. Derived operand
/// types are expected to provide the following:
///  * static IRObjectWithUseList *getUseList(IRValueT value);
///    - Provide the use list that is attached to the given value.
template <typename DerivedT, typename IRValueT>
class IROperand : public detail::IROperandBase {
public:
  IROperand(Operation *owner) : detail::IROperandBase(owner) {}
  IROperand(Operation *owner, IRValueT value)
      : detail::IROperandBase(owner), value(value) {
    insertIntoCurrent();
  }

  /// We support a move constructor so IROperand's can be in vectors, but this
  /// shouldn't be used by general clients.
  IROperand(IROperand &&other) : detail::IROperandBase(std::move(other)) {
    *this = std::move(other);
  }
  IROperand &operator=(IROperand &&other) {
    detail::IROperandBase::operator=(std::move(other));
    value = other.value;
    other.value = nullptr;
    if (value)
      insertIntoCurrent();
    return *this;
  }

  /// Two operands are equal if they have the same owner and the same operand
  /// number. They are stored inside of ops, so it is valid to compare their
  /// pointers to determine equality.
  bool operator==(const IROperand<DerivedT, IRValueT> &other) const {
    return this == &other;
  }
  bool operator!=(const IROperand<DerivedT, IRValueT> &other) const {
    return !(*this == other);
  }

  /// Return the current value being used by this operand.
  IRValueT get() const { return value; }

  /// Set the current value being used by this operand.
  void set(IRValueT newValue) {
    // It isn't worth optimizing for the case of switching operands on a single
    // value.
    removeFromCurrent();
    value = newValue;
    insertIntoCurrent();
  }

  /// Returns true if this operand contains the given value.
  bool is(IRValueT other) const { return value == other; }

  /// \brief Remove this use of the operand.
  void drop() {
    detail::IROperandBase::drop();
    value = nullptr;
  }

private:
  /// The value used as this operand. This can be null when in a 'dropAllUses'
  /// state.
  IRValueT value = {};

  /// Insert this operand into the given use list.
  void insertIntoCurrent() { insertInto(DerivedT::getUseList(value)); }
};

//===----------------------------------------------------------------------===//
// IRObjectWithUseList
//===----------------------------------------------------------------------===//

/// This class represents a single IR object that contains a use list.
template <typename OperandType>
class IRObjectWithUseList {
public:
  ~IRObjectWithUseList() {
    assert(use_empty() && "Cannot destroy a value that still has uses!");
  }

  /// Drop all uses of this object from their respective owners.
  void dropAllUses() {
    while (!use_empty())
      use_begin()->drop();
  }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  template <typename ValueT>
  void replaceAllUsesWith(ValueT &&newValue) {
    assert((!newValue || this != OperandType::getUseList(newValue)) &&
           "cannot RAUW a value with itself");
    while (!use_empty())
      use_begin()->set(newValue);
  }

  /// Shuffle the use-list chain according to the provided indices vector, which
  /// need to represent a valid shuffle. That is, a vector of unique integers in
  /// range [0, numUses - 1]. Users of this function need to guarantee the
  /// validity of the indices vector.
  void shuffleUseList(ArrayRef<unsigned> indices) {
    assert((size_t)std::distance(getUses().begin(), getUses().end()) ==
               indices.size() &&
           "indices vector expected to have a number of elements equal to the "
           "number of uses");
    SmallVector<detail::IROperandBase *> shuffled(indices.size());
    detail::IROperandBase *ptr = firstUse;
    for (size_t idx = 0; idx < indices.size();
         idx++, ptr = ptr->getNextOperandUsingThisValue())
      shuffled[indices[idx]] = ptr;

    initFirstUse(shuffled.front());
    auto *current = firstUse;
    for (auto &next : llvm::drop_begin(shuffled)) {
      current->linkTo(next);
      current = next;
    }
    current->linkTo(nullptr);
  }

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  using use_iterator = ValueUseIterator<OperandType>;
  using use_range = iterator_range<use_iterator>;

  use_iterator use_begin() const { return use_iterator(firstUse); }
  use_iterator use_end() const { return use_iterator(nullptr); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() const { return {use_begin(), use_end()}; }

  /// Returns true if this value has exactly one use.
  bool hasOneUse() const {
    return firstUse && firstUse->getNextOperandUsingThisValue() == nullptr;
  }

  /// Returns true if this value has no uses.
  bool use_empty() const { return firstUse == nullptr; }

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OperandType>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() const { return user_iterator(use_begin()); }
  user_iterator user_end() const { return user_iterator(use_end()); }

  /// Returns a range of all users.
  user_range getUsers() const { return {user_begin(), user_end()}; }

protected:
  IRObjectWithUseList() = default;

  /// Return the first operand that is using this value, for use by custom
  /// use/def iterators.
  OperandType *getFirstUse() const { return (OperandType *)firstUse; }

private:
  /// Set use as the first use of the chain.
  void initFirstUse(detail::IROperandBase *use) {
    firstUse = use;
    firstUse->initChainWithUse(&firstUse);
  }

  detail::IROperandBase *firstUse = nullptr;

  /// Allow access to `firstUse`.
  friend detail::IROperandBase;
};

//===----------------------------------------------------------------------===//
// ValueUseIterator
//===----------------------------------------------------------------------===//

/// An iterator class that allows for iterating over the uses of an IR operand
/// type.
template <typename OperandType>
class ValueUseIterator
    : public llvm::iterator_facade_base<ValueUseIterator<OperandType>,
                                        std::forward_iterator_tag,
                                        OperandType> {
public:
  ValueUseIterator(detail::IROperandBase *use = nullptr) : current(use) {}

  /// Returns the operation that owns this use.
  Operation *getUser() const { return current->getOwner(); }

  /// Returns the current operands.
  OperandType *getOperand() const { return (OperandType *)current; }
  OperandType &operator*() const { return *getOperand(); }

  using llvm::iterator_facade_base<ValueUseIterator<OperandType>,
                                   std::forward_iterator_tag,
                                   OperandType>::operator++;
  ValueUseIterator &operator++() {
    assert(current && "incrementing past end()!");
    current = (OperandType *)current->getNextOperandUsingThisValue();
    return *this;
  }

  bool operator==(const ValueUseIterator &rhs) const {
    return current == rhs.current;
  }

protected:
  detail::IROperandBase *current;
};

//===----------------------------------------------------------------------===//
// ValueUserIterator
//===----------------------------------------------------------------------===//

/// An iterator over the users of an IRObject. This is a wrapper iterator around
/// a specific use iterator.
template <typename UseIteratorT, typename OperandType>
class ValueUserIterator final
    : public llvm::mapped_iterator_base<
          ValueUserIterator<UseIteratorT, OperandType>, UseIteratorT,
          Operation *> {
public:
  using llvm::mapped_iterator_base<ValueUserIterator<UseIteratorT, OperandType>,
                                   UseIteratorT,
                                   Operation *>::mapped_iterator_base;

  /// Map the element to the iterator result type.
  Operation *mapElement(OperandType &value) const { return value.getOwner(); }

  /// Provide access to the underlying operation.
  Operation *operator->() { return **this; }
};

} // namespace mlir

#endif
