//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matching over MLIR. This mechanism is inspired by LLVM's
// include/llvm/IR/PatternMatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MATCHERS_H
#define MLIR_IR_MATCHERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {

namespace detail {

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <
    typename AttrClass,
    // Require AttrClass to be a derived class from Attribute and get its
    // value type
    typename ValueType = typename std::enable_if_t<
        std::is_base_of<Attribute, AttrClass>::value, AttrClass>::ValueType,
    // Require the ValueType is not void
    typename = std::enable_if_t<!std::is_void<ValueType>::value>>
struct attr_value_binder {
  ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType *bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    if (auto intAttr = llvm::dyn_cast<AttrClass>(attr)) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches operations that have the `ConstantLike` trait.
struct constant_op_matcher {
  bool match(Operation *op) { return op->hasTrait<OpTrait::ConstantLike>(); }
};

/// The matcher that matches operations that have the specified op name.
struct NameOpMatcher {
  NameOpMatcher(StringRef name) : name(name) {}
  bool match(Operation *op) { return op->getName().getStringRef() == name; }

  StringRef name;
};

/// The matcher that matches operations that have the specified attribute name.
struct AttrOpMatcher {
  AttrOpMatcher(StringRef attrName) : attrName(attrName) {}
  bool match(Operation *op) { return op->hasAttr(attrName); }

  StringRef attrName;
};

/// The matcher that matches operations that have the `ConstantLike` trait, and
/// binds the folded attribute value.
template <typename AttrT>
struct constant_op_binder {
  AttrT *bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT *bind_value) : bind_value(bind_value) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  constant_op_binder() : bind_value(nullptr) {}

  bool match(Operation *op) {
    if (!op->hasTrait<OpTrait::ConstantLike>())
      return false;

    // Fold the constant to an attribute.
    SmallVector<OpFoldResult, 1> foldedOp;
    LogicalResult result = op->fold(/*operands=*/std::nullopt, foldedOp);
    (void)result;
    assert(succeeded(result) && "expected ConstantLike op to be foldable");

    if (auto attr = llvm::dyn_cast<AttrT>(foldedOp.front().get<Attribute>())) {
      if (bind_value)
        *bind_value = attr;
      return true;
    }
    return false;
  }
};

/// A matcher that matches operations that implement the
/// `InferIntRangeInterface` interface, and binds the inferred range.
struct infer_int_range_op_binder {
  IntegerValueRange *bind_value;

  explicit infer_int_range_op_binder(IntegerValueRange *bind_value)
      : bind_value(bind_value) {}

  bool match(Operation *op) {
    auto inferIntRangeOp = dyn_cast<InferIntRangeInterface>(op);
    if (!inferIntRangeOp)
      return false;

    // Set the range of all integer operands to the maximal range.
    SmallVector<IntegerValueRange> argRanges =
        llvm::map_to_vector(op->getOperands(), IntegerValueRange::getMaxRange);

    // Infer the result result range if possible.
    bool matched = false;
    auto setResultRanges = [&](Value value,
                               const IntegerValueRange &argRanges) {
      if (argRanges.isUninitialized())
        return;
      if (value != op->getResult(0))
        return;
      *bind_value = argRanges;
      matched = true;
    };
    inferIntRangeOp.inferResultRangesFromOptional(argRanges, setResultRanges);
    return matched;
  }
};

/// The matcher that matches operations that have the specified attribute
/// name, and binds the attribute value.
template <typename AttrT>
struct AttrOpBinder {
  /// Creates a matcher instance that binds the attribute value to
  /// bind_value if match succeeds.
  AttrOpBinder(StringRef attrName, AttrT *bindValue)
      : attrName(attrName), bindValue(bindValue) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  AttrOpBinder(StringRef attrName) : attrName(attrName), bindValue(nullptr) {}

  bool match(Operation *op) {
    if (auto attr = op->getAttrOfType<AttrT>(attrName)) {
      if (bindValue)
        *bindValue = attr;
      return true;
    }
    return false;
  }
  StringRef attrName;
  AttrT *bindValue;
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// float Attribute or Operation and binds the constant float value.
struct constant_float_value_binder {
  FloatAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_float_value_binder(FloatAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    attr_value_binder<FloatAttr> matcher(bind_value);
    if (matcher.match(attr))
      return true;

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
      return matcher.match(splatAttr.getSplatValue<Attribute>());

    return false;
  }

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;

    Type type = op->getResult(0).getType();
    if (isa<FloatType, VectorType, RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat float value that fulfills a predicate.
struct constant_float_predicate_matcher {
  bool (*predicate)(const APFloat &);

  bool match(Attribute attr) {
    APFloat value(APFloat::Bogus());
    return constant_float_value_binder(&value).match(attr) && predicate(value);
  }

  bool match(Operation *op) {
    APFloat value(APFloat::Bogus());
    return constant_float_value_binder(&value).match(op) && predicate(value);
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer Attribute or Operation and binds the constant integer value.
struct constant_int_value_binder {
  IntegerAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_value_binder(IntegerAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    attr_value_binder<IntegerAttr> matcher(bind_value);
    if (matcher.match(attr))
      return true;

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
      return matcher.match(splatAttr.getSplatValue<Attribute>());

    return false;
  }

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;

    Type type = op->getResult(0).getType();
    if (isa<IntegerType, IndexType, VectorType, RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat integer value that fulfills a predicate.
struct constant_int_predicate_matcher {
  bool (*predicate)(const APInt &);

  bool match(Attribute attr) {
    APInt value;
    return constant_int_value_binder(&value).match(attr) && predicate(value);
  }

  bool match(Operation *op) {
    APInt value;
    return constant_int_value_binder(&value).match(op) && predicate(value);
  }
};

/// A matcher that matches a given a constant scalar / vector splat / tensor
/// splat integer value or a constant integer range that fulfills a predicate.
struct constant_int_range_predicate_matcher {
  bool (*predicate)(const ConstantIntRanges &);

  bool match(Attribute attr) {
    APInt value;
    return constant_int_value_binder(&value).match(attr) &&
           predicate(ConstantIntRanges::constant(value));
  }

  bool match(Operation *op) {
    // Try to match a constant integer value first.
    APInt value;
    if (constant_int_value_binder(&value).match(op))
      return predicate(ConstantIntRanges::constant(value));

    // Otherwise, try to match an operation that implements the
    // `InferIntRangeInterface` interface.
    IntegerValueRange range;
    return infer_int_range_op_binder(&range).match(op) &&
           predicate(range.getValue());
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpClass>
struct op_matcher {
  bool match(Operation *op) { return isa<OpClass>(op); }
};

/// Trait to check whether T provides a 'match' method with type
/// `MatchTarget` (Value, Operation, or Attribute).
template <typename T, typename MatchTarget>
using has_compatible_matcher_t =
    decltype(std::declval<T>().match(std::declval<MatchTarget>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_compatible_matcher_t,
                                   MatcherClass, Value>::value,
                 bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  return matcher.match(op->getOperand(idx));
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_compatible_matcher_t,
                                   MatcherClass, Operation *>::value,
                 bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  if (auto *defOp = op->getOperand(idx).getDefiningOp())
    return matcher.match(defOp);
  return false;
}

/// Terminal matcher, always returns true.
struct AnyValueMatcher {
  bool match(Value op) const { return true; }
};

/// Terminal matcher, always returns true.
struct AnyCapturedValueMatcher {
  Value *what;
  AnyCapturedValueMatcher(Value *what) : what(what) {}
  bool match(Value op) const {
    *what = op;
    return true;
  }
};

/// Binds to a specific value and matches it.
struct PatternMatcherValue {
  PatternMatcherValue(Value val) : value(val) {}
  bool match(Value val) const { return val == value; }
  Value value;
};

template <typename TupleT, class CallbackT, std::size_t... Is>
constexpr void enumerateImpl(TupleT &&tuple, CallbackT &&callback,
                             std::index_sequence<Is...>) {

  (callback(std::integral_constant<std::size_t, Is>{}, std::get<Is>(tuple)),
   ...);
}

template <typename... Tys, typename CallbackT>
constexpr void enumerate(std::tuple<Tys...> &tuple, CallbackT &&callback) {
  detail::enumerateImpl(tuple, std::forward<CallbackT>(callback),
                        std::make_index_sequence<sizeof...(Tys)>{});
}

/// RecursivePatternMatcher that composes.
template <typename OpType, typename... OperandMatchers>
struct RecursivePatternMatcher {
  RecursivePatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  bool match(Operation *op) {
    if (!isa<OpType>(op) || op->getNumOperands() != sizeof...(OperandMatchers))
      return false;
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher);
    });
    return res;
  }
  std::tuple<OperandMatchers...> operandMatchers;
};

} // namespace detail

/// Matches a constant foldable operation.
inline detail::constant_op_matcher m_Constant() {
  return detail::constant_op_matcher();
}

/// Matches a named attribute operation.
inline detail::AttrOpMatcher m_Attr(StringRef attrName) {
  return detail::AttrOpMatcher(attrName);
}

/// Matches a named operation.
inline detail::NameOpMatcher m_Op(StringRef opName) {
  return detail::NameOpMatcher(opName);
}

/// Matches a value from a constant foldable operation and writes the value to
/// bind_value.
template <typename AttrT>
inline detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return detail::constant_op_binder<AttrT>(bind_value);
}

/// Matches a named attribute operation and writes the value to bind_value.
template <typename AttrT>
inline detail::AttrOpBinder<AttrT> m_Attr(StringRef attrName,
                                          AttrT *bindValue) {
  return detail::AttrOpBinder<AttrT>(attrName, bindValue);
}

/// Matches a constant scalar / vector splat / tensor splat float (both positive
/// and negative) zero.
inline detail::constant_float_predicate_matcher m_AnyZeroFloat() {
  return {[](const APFloat &value) { return value.isZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float positive zero.
inline detail::constant_float_predicate_matcher m_PosZeroFloat() {
  return {[](const APFloat &value) { return value.isPosZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float negative zero.
inline detail::constant_float_predicate_matcher m_NegZeroFloat() {
  return {[](const APFloat &value) { return value.isNegZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float ones.
inline detail::constant_float_predicate_matcher m_OneFloat() {
  return {[](const APFloat &value) {
    return APFloat(value.getSemantics(), 1) == value;
  }};
}

/// Matches a constant scalar / vector splat / tensor splat float positive
/// infinity.
inline detail::constant_float_predicate_matcher m_PosInfFloat() {
  return {[](const APFloat &value) {
    return !value.isNegative() && value.isInfinity();
  }};
}

/// Matches a constant scalar / vector splat / tensor splat float negative
/// infinity.
inline detail::constant_float_predicate_matcher m_NegInfFloat() {
  return {[](const APFloat &value) {
    return value.isNegative() && value.isInfinity();
  }};
}

/// Matches a constant scalar / vector splat / tensor splat integer zero.
inline detail::constant_int_predicate_matcher m_Zero() {
  return {[](const APInt &value) { return 0 == value; }};
}

/// Matches a constant scalar / vector splat / tensor splat integer that is any
/// non-zero value.
inline detail::constant_int_predicate_matcher m_NonZero() {
  return {[](const APInt &value) { return 0 != value; }};
}

/// Matches a constant scalar / vector splat / tensor splat integer or a
/// unsigned integer range that does not contain zero. Note that this matcher
/// interprets the target value as an unsigned integer.
inline detail::constant_int_range_predicate_matcher m_IntRangeWithoutZeroU() {
  return {[](const ConstantIntRanges &range) { return range.umin().ugt(0); }};
}

/// Matches a constant scalar / vector splat / tensor splat integer or a
/// signed integer range that does not contain zero. Note that this matcher
/// interprets the target value as a signed integer.
inline detail::constant_int_range_predicate_matcher m_IntRangeWithoutZeroS() {
  return {[](const ConstantIntRanges &range) {
    return range.smin().sgt(0) || range.smax().slt(0);
  }};
}

/// Matches a constant scalar / vector splat / tensor splat integer or a
/// signed integer range that does not contain minus one. Note
/// that this matcher interprets the target value as a signed integer.
inline detail::constant_int_range_predicate_matcher m_IntRangeWithoutNegOneS() {
  return {[](const ConstantIntRanges &range) {
    return range.smin().sgt(-1) || range.smax().slt(-1);
  }};
}

/// Matches a constant scalar / vector splat / tensor splat integer one.
inline detail::constant_int_predicate_matcher m_One() {
  return {[](const APInt &value) { return 1 == value; }};
}

/// Matches the given OpClass.
template <typename OpClass>
inline detail::op_matcher<OpClass> m_Op() {
  return detail::op_matcher<OpClass>();
}

/// Entry point for matching a pattern over a Value.
template <typename Pattern>
inline bool matchPattern(Value value, const Pattern &pattern) {
  assert(value);
  // TODO: handle other cases
  if (auto *op = value.getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(Operation *op, const Pattern &pattern) {
  assert(op);
  return const_cast<Pattern &>(pattern).match(op);
}

/// Entry point for matching a pattern over an Attribute. Returns `false`
/// when `attr` is null.
template <typename Pattern>
inline bool matchPattern(Attribute attr, const Pattern &pattern) {
  static_assert(llvm::is_detected<detail::has_compatible_matcher_t, Pattern,
                                  Attribute>::value,
                "Pattern does not support matching Attributes");
  if (!attr)
    return false;
  return const_cast<Pattern &>(pattern).match(attr);
}

/// Matches a constant holding a scalar/vector/tensor float (splat) and
/// writes the float value to bind_value.
inline detail::constant_float_value_binder
m_ConstantFloat(FloatAttr::ValueType *bind_value) {
  return detail::constant_float_value_binder(bind_value);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline detail::constant_int_value_binder
m_ConstantInt(IntegerAttr::ValueType *bind_value) {
  return detail::constant_int_value_binder(bind_value);
}

template <typename OpType, typename... Matchers>
auto m_Op(Matchers... matchers) {
  return detail::RecursivePatternMatcher<OpType, Matchers...>(matchers...);
}

namespace matchers {
inline auto m_Any() { return detail::AnyValueMatcher(); }
inline auto m_Any(Value *val) { return detail::AnyCapturedValueMatcher(val); }
inline auto m_Val(Value v) { return detail::PatternMatcherValue(v); }
} // namespace matchers

} // namespace mlir

#endif // MLIR_IR_MATCHERS_H
