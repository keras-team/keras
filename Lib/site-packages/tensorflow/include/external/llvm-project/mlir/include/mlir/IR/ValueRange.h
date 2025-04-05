//===- ValueRange.h - Indexed Value-Iterators Range Classes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ValueRange related classes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VALUERANGE_H
#define MLIR_IR_VALUERANGE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/Sequence.h"
#include <optional>

namespace mlir {
class ValueRange;
template <typename ValueRangeT>
class ValueTypeRange;
class TypeRangeRange;
template <typename ValueIteratorT>
class ValueTypeIterator;
class OperandRangeRange;
class MutableOperandRangeRange;

//===----------------------------------------------------------------------===//
// Operation Value-Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperandRange

/// This class implements the operand iterators for the Operation class.
class OperandRange final : public llvm::detail::indexed_accessor_range_base<
                               OperandRange, OpOperand *, Value, Value, Value> {
public:
  using RangeBaseT::RangeBaseT;

  /// Returns the types of the values within this range.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<OperandRange>;
  type_range getTypes() const;
  type_range getType() const;

  /// Return the operand index of the first element of this range. The range
  /// must not be empty.
  unsigned getBeginOperandIndex() const;

  /// Split this range into a set of contiguous subranges using the given
  /// elements attribute, which contains the sizes of the sub ranges.
  OperandRangeRange split(DenseI32ArrayAttr segmentSizes) const;

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OpOperand *offset_base(OpOperand *object, ptrdiff_t index) {
    return object + index;
  }
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Value dereference_iterator(OpOperand *object, ptrdiff_t index) {
    return object[index].get();
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// OperandRangeRange

/// This class represents a contiguous range of operand ranges, e.g. from a
/// VariadicOfVariadic operand group.
class OperandRangeRange final
    : public llvm::indexed_accessor_range<
          OperandRangeRange, std::pair<OpOperand *, Attribute>, OperandRange,
          OperandRange, OperandRange> {
  using OwnerT = std::pair<OpOperand *, Attribute>;
  using RangeBaseT =
      llvm::indexed_accessor_range<OperandRangeRange, OwnerT, OperandRange,
                                   OperandRange, OperandRange>;

public:
  using RangeBaseT::RangeBaseT;

  /// Returns the range of types of the values within this range.
  TypeRangeRange getTypes() const;
  TypeRangeRange getType() const;

  /// Construct a range given a parent set of operands, and an I32 elements
  /// attribute containing the sizes of the sub ranges.
  OperandRangeRange(OperandRange operands, Attribute operandSegments);

  /// Flatten all of the sub ranges into a single contiguous operand range.
  OperandRange join() const;

private:
  /// See `llvm::indexed_accessor_range` for details.
  static OperandRange dereference(const OwnerT &object, ptrdiff_t index);

  /// Allow access to `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// MutableOperandRange

/// This class provides a mutable adaptor for a range of operands. It allows for
/// setting, inserting, and erasing operands from the given range.
class MutableOperandRange {
public:
  /// A pair of a named attribute corresponding to an operand segment attribute,
  /// and the index within that attribute. The attribute should correspond to a
  /// dense i32 array attr.
  using OperandSegment = std::pair<unsigned, NamedAttribute>;

  /// Construct a new mutable range from the given operand, operand start index,
  /// and range length. `operandSegments` is an optional set of operand segments
  /// to be updated when mutating the operand list.
  MutableOperandRange(Operation *owner, unsigned start, unsigned length,
                      ArrayRef<OperandSegment> operandSegments = std::nullopt);
  MutableOperandRange(Operation *owner);

  /// Construct a new mutable range for the given OpOperand.
  MutableOperandRange(OpOperand &opOperand);

  /// Slice this range into a sub range, with the additional operand segment.
  MutableOperandRange
  slice(unsigned subStart, unsigned subLen,
        std::optional<OperandSegment> segment = std::nullopt) const;

  /// Append the given values to the range.
  void append(ValueRange values);

  /// Assign this range to the given values.
  void assign(ValueRange values);

  /// Assign the range to the given value.
  void assign(Value value);

  /// Erase the operands within the given sub-range.
  void erase(unsigned subStart, unsigned subLen = 1);

  /// Clear this range and erase all of the operands.
  void clear();

  /// Returns the current size of the range.
  unsigned size() const { return length; }

  /// Returns if the current range is empty.
  bool empty() const { return size() == 0; }

  /// Explicit conversion to an OperandRange.
  OperandRange getAsOperandRange() const;

  /// Allow implicit conversion to an OperandRange.
  operator OperandRange() const;

  /// Allow implicit conversion to a MutableArrayRef.
  operator MutableArrayRef<OpOperand>() const;

  /// Returns the owning operation.
  Operation *getOwner() const { return owner; }

  /// Split this range into a set of contiguous subranges using the given
  /// elements attribute, which contains the sizes of the sub ranges.
  MutableOperandRangeRange split(NamedAttribute segmentSizes) const;

  /// Returns the OpOperand at the given index.
  OpOperand &operator[](unsigned index) const;

  /// Iterators enumerate OpOperands.
  MutableArrayRef<OpOperand>::iterator begin() const;
  MutableArrayRef<OpOperand>::iterator end() const;

private:
  /// Update the length of this range to the one provided.
  void updateLength(unsigned newLength);

  /// The owning operation of this range.
  Operation *owner;

  /// The start index of the operand range within the owner operand list, and
  /// the length starting from `start`.
  unsigned start, length;

  /// Optional set of operand segments that should be updated when mutating the
  /// length of this range.
  SmallVector<OperandSegment, 1> operandSegments;
};

//===----------------------------------------------------------------------===//
// MutableOperandRangeRange

/// This class represents a contiguous range of mutable operand ranges, e.g.
/// from a VariadicOfVariadic operand group.
class MutableOperandRangeRange final
    : public llvm::indexed_accessor_range<
          MutableOperandRangeRange,
          std::pair<MutableOperandRange, NamedAttribute>, MutableOperandRange,
          MutableOperandRange, MutableOperandRange> {
  using OwnerT = std::pair<MutableOperandRange, NamedAttribute>;
  using RangeBaseT =
      llvm::indexed_accessor_range<MutableOperandRangeRange, OwnerT,
                                   MutableOperandRange, MutableOperandRange,
                                   MutableOperandRange>;

public:
  using RangeBaseT::RangeBaseT;

  /// Construct a range given a parent set of operands, and an I32 tensor
  /// elements attribute containing the sizes of the sub ranges.
  MutableOperandRangeRange(const MutableOperandRange &operands,
                           NamedAttribute operandSegmentAttr);

  /// Flatten all of the sub ranges into a single contiguous mutable operand
  /// range.
  MutableOperandRange join() const;

  /// Allow implicit conversion to an OperandRangeRange.
  operator OperandRangeRange() const;

private:
  /// See `llvm::indexed_accessor_range` for details.
  static MutableOperandRange dereference(const OwnerT &object, ptrdiff_t index);

  /// Allow access to `dereference_iterator`.
  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// ResultRange

/// This class implements the result iterators for the Operation class.
class ResultRange final
    : public llvm::detail::indexed_accessor_range_base<
          ResultRange, detail::OpResultImpl *, OpResult, OpResult, OpResult> {
public:
  using RangeBaseT::RangeBaseT;
  ResultRange(OpResult result);

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  /// Returns the types of the values within this range.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<ResultRange>;
  type_range getTypes() const;
  type_range getType() const;

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  class UseIterator;
  using use_iterator = UseIterator;
  using use_range = iterator_range<use_iterator>;

  /// Returns a range of all uses of results within this range, which is useful
  /// for iterating over all uses.
  use_range getUses() const;
  use_iterator use_begin() const;
  use_iterator use_end() const;

  /// Returns true if no results in this range have uses.
  bool use_empty() const {
    return llvm::all_of(*this,
                        [](OpResult result) { return result.use_empty(); });
  }

  /// Replace all uses of results of this range with the provided 'values'. The
  /// size of `values` must match the size of this range.
  template <typename ValuesT>
  std::enable_if_t<!std::is_convertible<ValuesT, Operation *>::value>
  replaceAllUsesWith(ValuesT &&values) {
    assert(static_cast<size_t>(std::distance(values.begin(), values.end())) ==
               size() &&
           "expected 'values' to correspond 1-1 with the number of results");

    for (auto it : llvm::zip(*this, values))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }

  /// Replace all uses of results of this range with results of 'op'.
  void replaceAllUsesWith(Operation *op);

  /// Replace uses of results of this range with the provided 'values' if the
  /// given callback returns true. The size of `values` must match the size of
  /// this range.
  template <typename ValuesT>
  std::enable_if_t<!std::is_convertible<ValuesT, Operation *>::value>
  replaceUsesWithIf(ValuesT &&values,
                    function_ref<bool(OpOperand &)> shouldReplace) {
    assert(static_cast<size_t>(std::distance(values.begin(), values.end())) ==
               size() &&
           "expected 'values' to correspond 1-1 with the number of results");

    for (auto it : llvm::zip(*this, values))
      std::get<0>(it).replaceUsesWithIf(std::get<1>(it), shouldReplace);
  }

  /// Replace uses of results of this range with results of `op` if the given
  /// callback returns true.
  void replaceUsesWithIf(Operation *op,
                         function_ref<bool(OpOperand &)> shouldReplace);

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  /// Returns a range of all users.
  user_range getUsers();
  user_iterator user_begin();
  user_iterator user_end();

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static detail::OpResultImpl *offset_base(detail::OpResultImpl *object,
                                           ptrdiff_t index) {
    return object->getNextResultAtOffset(index);
  }
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OpResult dereference_iterator(detail::OpResultImpl *object,
                                       ptrdiff_t index) {
    return offset_base(object, index);
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

/// This class implements a use iterator for a range of operation results.
/// This iterates over all uses of all results within the given result range.
class ResultRange::UseIterator final
    : public llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                        OpOperand> {
public:
  /// Initialize the UseIterator. Specify `end` to return iterator to last
  /// use, otherwise this is an iterator to the first use.
  explicit UseIterator(ResultRange results, bool end = false);

  using llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                   OpOperand>::operator++;
  UseIterator &operator++();
  OpOperand *operator->() const { return use.getOperand(); }
  OpOperand &operator*() const { return *use.getOperand(); }

  bool operator==(const UseIterator &rhs) const { return use == rhs.use; }
  bool operator!=(const UseIterator &rhs) const { return !(*this == rhs); }

private:
  void skipOverResultsWithNoUsers();

  /// The range of results being iterated over.
  ResultRange::iterator it, endIt;
  /// The use of the result.
  Value::use_iterator use;
};

//===----------------------------------------------------------------------===//
// ValueRange

/// This class provides an abstraction over the different types of ranges over
/// Values. In many cases, this prevents the need to explicitly materialize a
/// SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class ValueRange final
    : public llvm::detail::indexed_accessor_range_base<
          ValueRange,
          PointerUnion<const Value *, OpOperand *, detail::OpResultImpl *>,
          Value, Value, Value> {
public:
  /// The type representing the owner of a ValueRange. This is either a list of
  /// values, operands, or results.
  using OwnerT =
      PointerUnion<const Value *, OpOperand *, detail::OpResultImpl *>;

  using RangeBaseT::RangeBaseT;

  template <typename Arg,
            typename = std::enable_if_t<
                std::is_constructible<ArrayRef<Value>, Arg>::value &&
                !std::is_convertible<Arg, Value>::value>>
  ValueRange(Arg &&arg) : ValueRange(ArrayRef<Value>(std::forward<Arg>(arg))) {}
  ValueRange(const Value &value) : ValueRange(&value, /*count=*/1) {}
  ValueRange(const std::initializer_list<Value> &values)
      : ValueRange(ArrayRef<Value>(values)) {}
  ValueRange(iterator_range<OperandRange::iterator> values)
      : ValueRange(OperandRange(values)) {}
  ValueRange(iterator_range<ResultRange::iterator> values)
      : ValueRange(ResultRange(values)) {}
  ValueRange(ArrayRef<BlockArgument> values)
      : ValueRange(ArrayRef<Value>(values.data(), values.size())) {}
  ValueRange(ArrayRef<Value> values = std::nullopt);
  ValueRange(OperandRange values);
  ValueRange(ResultRange values);

  /// Returns the types of the values within this range.
  using type_iterator = ValueTypeIterator<iterator>;
  using type_range = ValueTypeRange<ValueRange>;
  type_range getTypes() const;
  type_range getType() const;

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(const OwnerT &owner, ptrdiff_t index);
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Value dereference_iterator(const OwnerT &owner, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

} // namespace mlir

#endif // MLIR_IR_VALUERANGE_H
