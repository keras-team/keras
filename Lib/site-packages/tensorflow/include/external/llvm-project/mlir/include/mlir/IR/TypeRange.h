//===- TypeRange.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypeRange and ValueTypeRange classes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPERANGE_H
#define MLIR_IR_TYPERANGE_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/Sequence.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// TypeRange

/// This class provides an abstraction over the various different ranges of
/// value types. In many cases, this prevents the need to explicitly materialize
/// a SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class TypeRange : public llvm::detail::indexed_accessor_range_base<
                      TypeRange,
                      llvm::PointerUnion<const Value *, const Type *,
                                         OpOperand *, detail::OpResultImpl *>,
                      Type, Type, Type> {
public:
  using RangeBaseT::RangeBaseT;
  TypeRange(ArrayRef<Type> types = std::nullopt);
  explicit TypeRange(OperandRange values);
  explicit TypeRange(ResultRange values);
  explicit TypeRange(ValueRange values);
  template <typename ValueRangeT>
  TypeRange(ValueTypeRange<ValueRangeT> values)
      : TypeRange(ValueRange(ValueRangeT(values.begin().getCurrent(),
                                         values.end().getCurrent()))) {}
  template <typename Arg, typename = std::enable_if_t<std::is_constructible<
                              ArrayRef<Type>, Arg>::value>>
  TypeRange(Arg &&arg) : TypeRange(ArrayRef<Type>(std::forward<Arg>(arg))) {}
  TypeRange(std::initializer_list<Type> types)
      : TypeRange(ArrayRef<Type>(types)) {}

private:
  /// The owner of the range is either:
  /// * A pointer to the first element of an array of values.
  /// * A pointer to the first element of an array of types.
  /// * A pointer to the first element of an array of operands.
  /// * A pointer to the first element of an array of results.
  using OwnerT = llvm::PointerUnion<const Value *, const Type *, OpOperand *,
                                    detail::OpResultImpl *>;

  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(OwnerT object, ptrdiff_t index);
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static Type dereference_iterator(OwnerT object, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

/// Make TypeRange hashable.
inline ::llvm::hash_code hash_value(TypeRange arg) {
  return ::llvm::hash_combine_range(arg.begin(), arg.end());
}

/// Emit a type range to the given output stream.
inline raw_ostream &operator<<(raw_ostream &os, const TypeRange &types) {
  llvm::interleaveComma(types, os);
  return os;
}

//===----------------------------------------------------------------------===//
// TypeRangeRange

using TypeRangeRangeIterator =
    llvm::mapped_iterator<llvm::iota_range<unsigned>::iterator,
                          std::function<TypeRange(unsigned)>>;

/// This class provides an abstraction for a range of TypeRange. This is useful
/// when accessing the types of a range of ranges, such as when using
/// OperandRangeRange.
class TypeRangeRange : public llvm::iterator_range<TypeRangeRangeIterator> {
public:
  template <typename RangeT>
  TypeRangeRange(const RangeT &range)
      : TypeRangeRange(llvm::seq<unsigned>(0, range.size()), range) {}

private:
  template <typename RangeT>
  TypeRangeRange(llvm::iota_range<unsigned> sizeRange, const RangeT &range)
      : llvm::iterator_range<TypeRangeRangeIterator>(
            {sizeRange.begin(), getRangeFn(range)},
            {sizeRange.end(), nullptr}) {}

  template <typename RangeT>
  static std::function<TypeRange(unsigned)> getRangeFn(const RangeT &range) {
    return [=](unsigned index) -> TypeRange { return TypeRange(range[index]); };
  }
};

//===----------------------------------------------------------------------===//
// ValueTypeRange

/// This class implements iteration on the types of a given range of values.
template <typename ValueIteratorT>
class ValueTypeIterator final
    : public llvm::mapped_iterator_base<ValueTypeIterator<ValueIteratorT>,
                                        ValueIteratorT, Type> {
public:
  using llvm::mapped_iterator_base<ValueTypeIterator<ValueIteratorT>,
                                   ValueIteratorT, Type>::mapped_iterator_base;

  /// Map the element to the iterator result type.
  Type mapElement(Value value) const { return value.getType(); }
};

/// This class implements iteration on the types of a given range of values.
template <typename ValueRangeT>
class ValueTypeRange final
    : public llvm::iterator_range<
          ValueTypeIterator<typename ValueRangeT::iterator>> {
public:
  using llvm::iterator_range<
      ValueTypeIterator<typename ValueRangeT::iterator>>::iterator_range;
  template <typename Container>
  ValueTypeRange(Container &&c) : ValueTypeRange(c.begin(), c.end()) {}

  /// Return the type at the given index.
  Type operator[](size_t index) const {
    assert(index < size() && "invalid index into type range");
    return *(this->begin() + index);
  }

  /// Return the size of this range.
  size_t size() const { return llvm::size(*this); }

  /// Return first type in the range.
  Type front() { return (*this)[0]; }

  /// Compare this range with another.
  template <typename OtherT>
  bool operator==(const OtherT &other) const {
    return llvm::size(*this) == llvm::size(other) &&
           std::equal(this->begin(), this->end(), other.begin());
  }
  template <typename OtherT>
  bool operator!=(const OtherT &other) const {
    return !(*this == other);
  }
};

template <typename RangeT>
inline bool operator==(ArrayRef<Type> lhs, const ValueTypeRange<RangeT> &rhs) {
  return lhs.size() == static_cast<size_t>(llvm::size(rhs)) &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

//===----------------------------------------------------------------------===//
// SubElements
//===----------------------------------------------------------------------===//

/// Enable TypeRange to be introspected for sub-elements.
template <>
struct AttrTypeSubElementHandler<TypeRange> {
  static void walk(TypeRange param, AttrTypeImmediateSubElementWalker &walker) {
    walker.walkRange(param);
  }
  static TypeRange replace(TypeRange param,
                           AttrSubElementReplacements &attrRepls,
                           TypeSubElementReplacements &typeRepls) {
    return typeRepls.take_front(param.size());
  }
};

} // namespace mlir

namespace llvm {

// Provide DenseMapInfo for TypeRange.
template <>
struct DenseMapInfo<mlir::TypeRange> {
  static mlir::TypeRange getEmptyKey() {
    return mlir::TypeRange(getEmptyKeyPointer(), 0);
  }

  static mlir::TypeRange getTombstoneKey() {
    return mlir::TypeRange(getTombstoneKeyPointer(), 0);
  }

  static unsigned getHashValue(mlir::TypeRange val) { return hash_value(val); }

  static bool isEqual(mlir::TypeRange lhs, mlir::TypeRange rhs) {
    if (isEmptyKey(rhs))
      return isEmptyKey(lhs);
    if (isTombstoneKey(rhs))
      return isTombstoneKey(lhs);
    return lhs == rhs;
  }

private:
  static const mlir::Type *getEmptyKeyPointer() {
    return DenseMapInfo<mlir::Type *>::getEmptyKey();
  }

  static const mlir::Type *getTombstoneKeyPointer() {
    return DenseMapInfo<mlir::Type *>::getTombstoneKey();
  }

  static bool isEmptyKey(mlir::TypeRange range) {
    if (const auto *type =
            llvm::dyn_cast_if_present<const mlir::Type *>(range.getBase()))
      return type == getEmptyKeyPointer();
    return false;
  }

  static bool isTombstoneKey(mlir::TypeRange range) {
    if (const auto *type =
            llvm::dyn_cast_if_present<const mlir::Type *>(range.getBase()))
      return type == getTombstoneKeyPointer();
    return false;
  }
};

} // namespace llvm

#endif // MLIR_IR_TYPERANGE_H
