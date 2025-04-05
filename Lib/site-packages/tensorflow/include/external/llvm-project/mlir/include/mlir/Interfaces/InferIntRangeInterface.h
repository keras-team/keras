//===- InferIntRangeInterface.h - Integer Range Inference --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the integer range inference interface
// defined in `InferIntRange.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERINTRANGEINTERFACE_H
#define MLIR_INTERFACES_INFERINTRANGEINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include <optional>

namespace mlir {
/// A set of arbitrary-precision integers representing bounds on a given integer
/// value. These bounds are inclusive on both ends, so
/// bounds of [4, 5] mean 4 <= x <= 5. Separate bounds are tracked for
/// the unsigned and signed interpretations of values in order to enable more
/// precice inference of the interplay between operations with signed and
/// unsigned semantics.
class ConstantIntRanges {
public:
  /// Bound umin <= (unsigned)x <= umax and smin <= signed(x) <= smax.
  /// Non-integer values should be bounded by APInts of bitwidth 0.
  ConstantIntRanges(const APInt &umin, const APInt &umax, const APInt &smin,
                    const APInt &smax)
      : uminVal(umin), umaxVal(umax), sminVal(smin), smaxVal(smax) {
    assert(uminVal.getBitWidth() == umaxVal.getBitWidth() &&
           umaxVal.getBitWidth() == sminVal.getBitWidth() &&
           sminVal.getBitWidth() == smaxVal.getBitWidth() &&
           "All bounds in the ranges must have the same bitwidth");
  }

  bool operator==(const ConstantIntRanges &other) const;

  /// The minimum value of an integer when it is interpreted as unsigned.
  const APInt &umin() const;

  /// The maximum value of an integer when it is interpreted as unsigned.
  const APInt &umax() const;

  /// The minimum value of an integer when it is interpreted as signed.
  const APInt &smin() const;

  /// The maximum value of an integer when it is interpreted as signed.
  const APInt &smax() const;

  /// Return the bitwidth that should be used for integer ranges describing
  /// `type`. For concrete integer types, this is their bitwidth, for `index`,
  /// this is the internal storage bitwidth of `index` attributes, and for
  /// non-integer types this is 0.
  static unsigned getStorageBitwidth(Type type);

  /// Create a `ConstantIntRanges` with the maximum bounds for the width
  /// `bitwidth`, that is - [0, uint_max(width)]/[sint_min(width),
  /// sint_max(width)].
  static ConstantIntRanges maxRange(unsigned bitwidth);

  /// Create a `ConstantIntRanges` with a constant value - that is, with the
  /// bounds [value, value] for both its signed interpretations.
  static ConstantIntRanges constant(const APInt &value);

  /// Create a `ConstantIntRanges` whose minimum is `min` and maximum is `max`
  /// with `isSigned` specifying if the min and max should be interpreted as
  /// signed or unsigned.
  static ConstantIntRanges range(const APInt &min, const APInt &max,
                                 bool isSigned);

  /// Create an `ConstantIntRanges` with the signed minimum and maximum equal
  /// to `smin` and `smax`, where the unsigned bounds are constructed from the
  /// signed ones if they correspond to a contigious range of bit patterns when
  /// viewed as unsigned values and are left at [0, int_max()] otherwise.
  static ConstantIntRanges fromSigned(const APInt &smin, const APInt &smax);

  /// Create an `ConstantIntRanges` with the unsigned minimum and maximum equal
  /// to `umin` and `umax` and the signed part equal to `umin` and `umax`
  /// unless the sign bit changes between the minimum and maximum.
  static ConstantIntRanges fromUnsigned(const APInt &umin, const APInt &umax);

  /// Returns the union (computed separately for signed and unsigned bounds)
  /// of this range and `other`.
  ConstantIntRanges rangeUnion(const ConstantIntRanges &other) const;

  /// Returns the intersection (computed separately for signed and unsigned
  /// bounds) of this range and `other`.
  ConstantIntRanges intersection(const ConstantIntRanges &other) const;

  /// If either the signed or unsigned interpretations of the range
  /// indicate that the value it bounds is a constant, return that constant
  /// value.
  std::optional<APInt> getConstantValue() const;

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const ConstantIntRanges &range);

private:
  APInt uminVal, umaxVal, sminVal, smaxVal;
};

raw_ostream &operator<<(raw_ostream &, const ConstantIntRanges &);

/// This lattice value represents the integer range of an SSA value.
class IntegerValueRange {
public:
  /// Create a maximal range ([0, uint_max(t)] / [int_min(t), int_max(t)])
  /// range that is used to mark the value as unable to be analyzed further,
  /// where `t` is the type of `value`.
  static IntegerValueRange getMaxRange(Value value);

  /// Create an integer value range lattice value.
  IntegerValueRange(ConstantIntRanges value) : value(std::move(value)) {}

  /// Create an integer value range lattice value.
  IntegerValueRange(std::optional<ConstantIntRanges> value = std::nullopt)
      : value(std::move(value)) {}

  /// Whether the range is uninitialized. This happens when the state hasn't
  /// been set during the analysis.
  bool isUninitialized() const { return !value.has_value(); }

  /// Get the known integer value range.
  const ConstantIntRanges &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  /// Compare two ranges.
  bool operator==(const IntegerValueRange &rhs) const {
    return value == rhs.value;
  }

  /// Compute the least upper bound of two ranges.
  static IntegerValueRange join(const IntegerValueRange &lhs,
                                const IntegerValueRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return IntegerValueRange{lhs.getValue().rangeUnion(rhs.getValue())};
  }

  /// Print the integer value range.
  void print(raw_ostream &os) const { os << value; }

private:
  /// The known integer value range.
  std::optional<ConstantIntRanges> value;
};

raw_ostream &operator<<(raw_ostream &, const IntegerValueRange &);

/// The type of the `setResultRanges` callback provided to ops implementing
/// InferIntRangeInterface. It should be called once for each integer result
/// value and be passed the ConstantIntRanges corresponding to that value.
using SetIntRangeFn =
    llvm::function_ref<void(Value, const ConstantIntRanges &)>;

/// Similar to SetIntRangeFn, but operating on IntegerValueRange lattice values.
/// This is the `setResultRanges` callback for the IntegerValueRange based
/// interface method.
using SetIntLatticeFn =
    llvm::function_ref<void(Value, const IntegerValueRange &)>;

class InferIntRangeInterface;

namespace intrange::detail {
/// Default implementation of `inferResultRanges` which dispatches to the
/// `inferResultRangesFromOptional`.
void defaultInferResultRanges(InferIntRangeInterface interface,
                              ArrayRef<IntegerValueRange> argRanges,
                              SetIntLatticeFn setResultRanges);

/// Default implementation of `inferResultRangesFromOptional` which dispatches
/// to the `inferResultRanges`.
void defaultInferResultRangesFromOptional(InferIntRangeInterface interface,
                                          ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRanges);
} // end namespace intrange::detail
} // end namespace mlir

#include "mlir/Interfaces/InferIntRangeInterface.h.inc"

#endif // MLIR_INTERFACES_INFERINTRANGEINTERFACE_H
