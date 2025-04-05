/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_REFERENCE_ELEMENT_H
#define STABLEHLO_REFERENCE_ELEMENT_H

#include <complex>
#include <variant>

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stablehlo {

/// Class to represent an element of a tensor. An Element object stores the
/// element type of the tensor and, depending on that element type, a constant
/// value of type integer, floating-paint, or complex type.
class Element {
 public:
  /// \name Constructors
  /// @{
  /// Initializes Element object with type `type` and value `value`. `type` must
  /// be an integer type.
  Element(Type type, APInt value);

  /// Initializes Element object with type `type` and value `value`. `type` must
  /// be a boolean type.
  Element(Type type, bool value);

  /// Initializes Element object with type `type` and value `value`. `type` must
  /// be a floating-point type of the same semantics as `value`.
  Element(Type type, APFloat value);

  /// Initializes Element object with type `type` and value `value`. `type` must
  /// be a complex type of the same semantics as `value`.
  Element(Type type, std::complex<APFloat> value);

  Element(const Element &other) = default;
  /// @}

  /// Assignment operator.
  Element &operator=(const Element &other) = default;

  /// Returns type of the Element object.
  Type getType() const { return type_; }

  /// Returns the underlying integer value stored in an Element object with
  /// integer type.
  APInt getIntegerValue() const;

  /// Returns the underlying boolean value stored in an Element object with
  /// bool type.
  bool getBooleanValue() const;

  /// Returns the underlying floating-point value stored in an Element object
  /// with floating-point type.
  APFloat getFloatValue() const;

  /// Returns the underlying complex value stored in an Element object with
  /// complex type.
  std::complex<APFloat> getComplexValue() const;

  /// Returns the implementation-defined bits of the underlying value.
  APInt toBits() const;

  /// Creates an Element from implementation-defined bits.
  static Element fromBits(Type type, APInt bits);

  /// Overloaded not (logical) operator.
  Element operator!() const;

  /// Overloaded inequality operator.
  Element operator!=(const Element &other) const;

  /// Overloaded and (bitwise) operator.
  Element operator&(const Element &other) const;

  /// Overloaded multiply operator.
  Element operator*(const Element &other) const;

  /// Overloaded add operator.
  Element operator+(const Element &other) const;

  /// Overloaded negate operator.
  Element operator-() const;

  /// Overloaded subtract operator.
  Element operator-(const Element &other) const;

  /// Overloaded divide operator.
  Element operator/(const Element &other) const;

  /// Overloaded less-than operator.
  Element operator<(const Element &other) const;

  /// Overloaded less-than-or-equal-to operator.
  Element operator<=(const Element &other) const;

  /// Overloaded equality operator.
  Element operator==(const Element &other) const;

  /// Overloaded greater-than operator.
  Element operator>(const Element &other) const;

  /// Overloaded greater-than-or-equal-to operator.
  Element operator>=(const Element &other) const;

  /// Overloaded xor (bitwise) operator.
  Element operator^(const Element &other) const;

  /// Overloaded or (bitwise) operator.
  Element operator|(const Element &other) const;

  /// Overloaded or (logical) operator.
  Element operator||(const Element &other) const;

  /// Overloaded not (bitwise) operator.
  Element operator~() const;

  /// Print utilities for Element objects.
  void print(raw_ostream &os, bool elideType = true) const;

  /// Print utilities for Element objects.
  void dump() const;

 private:
  Type type_;
  std::variant<APInt, bool, APFloat, std::pair<APFloat, APFloat>> value_;
};

/// Returns abs of Element object.
Element abs(const Element &e);

/// Returns atan2 of Element object.
Element atan2(const Element &e1, const Element &e2);

/// For floating point type T, checks if two normal values f1 and f2 are equal
/// within a set tolerance (default 0.0001).
/// For complex element type, checks if both real and imaginary parts are
/// individually equal modulo the tolerance.
Element areApproximatelyEqual(const Element &e1, const Element &e2,
                              APFloat tolerance = APFloat(0.0001));

/// Various flavors of bitcast conversion as defined in the specification.
Element bitcastConvertOneToOne(Type type, const Element &e);
SmallVector<Element> bitcastConvertOneToMany(Type type, const Element &e);
Element bitcastConvertManyToOne(Type type, ArrayRef<Element> es);

/// Returns cube root of Element object.
Element cbrt(const Element &e);

/// Returns ceil of Element object.
Element ceil(const Element &e);

/// Returns a complex type Element object.
Element complex(const Element &e1, const Element &e2);

/// Returns converted Element object.
Element convert(Type type, const Element &e);

/// Returns converted Element object of type `type` from source boolean `value`.
Element convert(Type type, bool value);

/// Returns converted Element object of type `type` from source signed integer
/// `value`. If the value cannot be exactly represented in the destination type,
/// then the behavior is TBD (#180).
Element convert(Type type, int64_t value);

/// Returns converted Element object of type `type` from source unsigned integer
/// `value`. If the value cannot be exactly represented in the destination type,
/// then the behavior is TBD (#180).
Element convert(Type type, uint64_t value);

/// Returns converted Element object of type `type` from source APFloat `value`.
/// If the value cannot be exactly represented in the destination type, then the
/// behavior is TBD (#180).
Element convert(Type type, APFloat value);

/// Returns converted Element object of type `type` from source APInt `value`.
/// If the value cannot be exactly represented in the destination type, then the
/// behavior is TBD (#180).
Element convert(Type type, APInt value, bool isSigned = false);

/// Returns converted Element object of type `type` from source APSInt `value`.
/// If the value cannot be exactly represented in the destination type, then the
/// behavior is TBD (#180).
Element convert(Type type, APSInt value);

/// Returns converted Element object of type `type` from source double `value`.
/// If the value cannot be exactly represented in the destination type, then the
/// behavior is TBD (#180).
Element convert(Type type, double value);

/// Returns converted Element object of type `type` from source complex<APFloat>
/// `value`. Only the real part of `value` is used to convert to non-complex
/// destination types. If the value (or the real part of `value`) cannot be
/// exactly represented in the complex destination type (or non-complex
/// destination type), then then the behavior is TBD (#180). If the real part of
/// `value` cannot be exactly represented in the non-complex destination type,
/// then the behavior is also TBD (#180).
Element convert(Type type, std::complex<APFloat> value);

/// Returns converted Element object of type `type` from source complex<double>
/// `value`. Only the real part of `value` is used to convert to non-complex
/// destination types. If the value (or the real part of `value`) cannot be
/// exactly represented in the complex destination type (or non-complex
/// destination type), then then the behavior is TBD (#180). If the real part of
/// `value` cannot be exactly represented in the non-complex destination type,
/// then the behavior is also TBD (#180).
Element convert(Type type, std::complex<double> value);

/// Returns Element object of type `type` from zero `value`
Element getZeroValueOfType(Type type);

/// Returns cosine of Element object.
Element cosine(const Element &e);

/// Returns exponential of Element object.
Element exponential(const Element &el);

/// Returns exponential_minus_one of Element object.
Element exponentialMinusOne(const Element &el);

/// Returns floor of Element object.
Element floor(const Element &e);

/// Returns the imaginary part extracted from the Element object with
/// floating-point or complex type.
Element imag(const Element &el);

/// Returns if the floating-point element object is finite.
Element isFinite(const Element &el);

/// Returns log of Element object.
Element log(const Element &el);

/// Returns log1p of Element object.
Element logPlusOne(const Element &el);

/// Returns logistic of Element object.
Element logistic(const Element &el);

/// Returns the maximum between two Element objects.
Element max(const Element &e1, const Element &e2);

/// Returns the minimum between two Element objects.
Element min(const Element &e1, const Element &e2);

/// Returns the population count of Element object.
Element popcnt(const Element &el);

/// Returns the exponentiation of first element to the power of second element.
Element power(const Element &e1, const Element &e2);

/// Returns the real part extracted from the Element object with floating-point
/// or complex type.
Element real(const Element &e);

/// Returns the Element object rounded to the specified precision type.
Element reducePrecision(const Element &el, int32_t exponentBits,
                        int32_t mantissaBits);

/// Returns the remainder for two Element objects.
Element rem(const Element &e1, const Element &e2);

/// Returns the value rounded to the nearest integer, breaking ties away from
/// zero, of Element object.
Element roundNearestAfz(const Element &el);

/// Returns the value rounded to nearest integer, breaking ties towards the
/// even, of Element object.
Element roundNearestEven(const Element &el);

/// Returns reverse square root of Element object.
Element rsqrt(const Element &e);

/// Returns left-shift of Element object e1 by e2.
Element shiftLeft(const Element &e1, const Element &e2);

/// Returns arithmetic right-shift of Element object e1 by e2.
Element shiftRightArithmetic(const Element &e1, const Element &e2);

/// Returns logical right-shift of Element object e1 by e2.
Element shiftRightLogical(const Element &e1, const Element &e2);

/// Returns sign of Element object.
Element sign(const Element &e);

/// Returns sine of Element object.
Element sine(const Element &e);

/// Returns square root of Element object.
Element sqrt(const Element &e);

/// Returns tan of Element object.
Element tan(const Element &e);

/// Returns tanh of Element object.
Element tanh(const Element &e);

/// Print utilities for Element objects.
inline raw_ostream &operator<<(raw_ostream &os, Element element) {
  element.print(os, /*elideType=*/true);
  return os;
}

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_ELEMENT_H
