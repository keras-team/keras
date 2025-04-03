/* Copyright 2023 The ml_dtypes Authors

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

#ifndef ML_DTYPES_INTN_H_
#define ML_DTYPES_INTN_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace ml_dtypes {

// Stores the n-bit integer value in the low n bits of a byte.  The upper
// bits are left unspecified and ignored.
template <int N, typename UnderlyingTy>
struct intN {
 private:
  UnderlyingTy v_;
  using SignedUnderlyingTy = std::make_signed_t<UnderlyingTy>;
  using UnsignedUnderlyingTy = std::make_unsigned_t<UnderlyingTy>;
  static constexpr int kUnderlyingBits =
      std::numeric_limits<UnsignedUnderlyingTy>::digits;

  static_assert(
      std::is_same_v<UnderlyingTy, uint8_t> ||
          std::is_same_v<UnderlyingTy, int8_t>,
      "The underyling type must be a signed or unsigned 8-bit integer.");

  // Mask the upper bits.
  static inline constexpr UnderlyingTy Mask(UnderlyingTy v) {
    return static_cast<UnsignedUnderlyingTy>(
               static_cast<UnsignedUnderlyingTy>(v) << (kUnderlyingBits - N)) >>
           (kUnderlyingBits - N);
  }

  // Mask the upper bits and sign-extend for signed types.
  static inline constexpr UnderlyingTy ExtendToFullWidth(UnderlyingTy v) {
    return static_cast<UnderlyingTy>(static_cast<UnderlyingTy>(v)
                                     << (kUnderlyingBits - N)) >>
           (kUnderlyingBits - N);
  }

  // Casts to the corresponding UnderlyingTy value.
  inline constexpr UnderlyingTy IntValue() const {
    return ExtendToFullWidth(v_);
  }

 public:
  constexpr intN() noexcept : v_(0) {}
  constexpr intN(const intN& other) noexcept = default;
  constexpr intN(intN&& other) noexcept = default;
  constexpr intN& operator=(const intN& other) = default;
  constexpr intN& operator=(intN&&) = default;

  explicit constexpr intN(UnderlyingTy val) : v_(Mask(val)) {}
  template <typename T>
  explicit constexpr intN(T t) : intN(static_cast<UnderlyingTy>(t)) {}

  using underlying_type = UnderlyingTy;
  static constexpr int bits = N;
  static constexpr int digits = std::is_signed_v<UnderlyingTy> ? N - 1 : N;
  static constexpr intN highest() { return intN((1 << digits) - 1); }
  static constexpr intN lowest() {
    return std::is_signed_v<UnderlyingTy> ? intN(1) << digits : intN(0);
  }

  template <typename T>
  explicit constexpr operator T() const {
    return static_cast<T>(IntValue());
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator std::optional<int64_t>() const {
    return static_cast<int64_t>(IntValue());
  }

  constexpr intN operator-() const { return intN(-v_); }
  constexpr intN operator+(const intN& other) const {
    return intN(v_ + other.v_);
  }
  constexpr intN operator-(const intN& other) const {
    return intN(v_ - other.v_);
  }
  constexpr intN operator*(const intN& other) const {
    return intN(v_ * other.v_);
  }
  constexpr intN operator/(const intN& other) const {
    return intN(IntValue() / other.IntValue());
  }
  constexpr intN operator%(const intN& other) const {
    return intN((IntValue() % other.IntValue()));
  }

  constexpr intN operator&(const intN& other) const {
    return intN(v_ & other.v_);
  }
  constexpr intN operator|(const intN& other) const {
    return intN(v_ | other.v_);
  }
  constexpr intN operator^(const intN& other) const {
    return intN(v_ ^ other.v_);
  }
  constexpr intN operator~() const { return intN(~v_); }
  constexpr intN operator>>(int amount) const {
    return intN(IntValue() >> amount);
  }
  constexpr intN operator<<(int amount) const { return intN(v_ << amount); }

  constexpr bool operator==(const intN& other) const {
    return Mask(v_) == Mask(other.v_);
  }
  constexpr bool operator!=(const intN& other) const {
    return Mask(v_) != Mask(other.v_);
  }
  constexpr bool operator<(const intN& other) const {
    return IntValue() < other.IntValue();
  }
  constexpr bool operator>(const intN& other) const {
    return IntValue() > other.IntValue();
  }
  constexpr bool operator<=(const intN& other) const {
    return IntValue() <= other.IntValue();
  }
  constexpr bool operator>=(const intN& other) const {
    return IntValue() >= other.IntValue();
  }

  constexpr bool operator==(int64_t other) const { return IntValue() == other; }
  constexpr bool operator!=(int64_t other) const { return IntValue() != other; }
  constexpr bool operator<(int64_t other) const { return IntValue() < other; }
  constexpr bool operator>(int64_t other) const { return IntValue() > other; }
  constexpr bool operator<=(int64_t other) const { return IntValue() <= other; }
  constexpr bool operator>=(int64_t other) const { return IntValue() >= other; }

  friend constexpr bool operator==(int64_t a, const intN& b) {
    return a == b.IntValue();
  }
  friend constexpr bool operator!=(int64_t a, const intN& b) {
    return a != b.IntValue();
  }
  friend constexpr bool operator<(int64_t a, const intN& b) {
    return a < b.IntValue();
  }
  friend constexpr bool operator>(int64_t a, const intN& b) {
    return a > b.IntValue();
  }
  friend constexpr bool operator<=(int64_t a, const intN& b) {
    return a <= b.IntValue();
  }
  friend constexpr bool operator>=(int64_t a, const intN& b) {
    return a >= b.IntValue();
  }

  constexpr intN& operator++() {
    v_ = Mask(v_ + 1);
    return *this;
  }

  constexpr intN operator++(int) {
    intN orig = *this;
    this->operator++();
    return orig;
  }

  constexpr intN& operator--() {
    v_ = Mask(v_ - 1);
    return *this;
  }

  constexpr intN operator--(int) {
    intN orig = *this;
    this->operator--();
    return orig;
  }

  constexpr intN& operator+=(const intN& other) {
    *this = *this + other;
    return *this;
  }
  constexpr intN& operator-=(const intN& other) {
    *this = *this - other;
    return *this;
  }
  constexpr intN& operator*=(const intN& other) {
    *this = *this * other;
    return *this;
  }
  constexpr intN& operator/=(const intN& other) {
    *this = *this / other;
    return *this;
  }
  constexpr intN& operator%=(const intN& other) {
    *this = *this % other;
    return *this;
  }
  constexpr intN& operator&=(const intN& other) {
    *this = *this & other;
    return *this;
  }
  constexpr intN& operator|=(const intN& other) {
    *this = *this | other;
    return *this;
  }
  constexpr intN& operator^=(const intN& other) {
    *this = *this ^ other;
    return *this;
  }
  constexpr intN& operator>>=(int amount) {
    *this = *this >> amount;
    return *this;
  }
  constexpr intN& operator<<=(int amount) {
    *this = *this << amount;
    return *this;
  }

  friend ::std::ostream& operator<<(::std::ostream& os, const intN& num) {
    os << static_cast<int16_t>(num);
    return os;
  }

  std::string ToString() const {
    std::ostringstream os;
    os << static_cast<int16_t>(*this);
    return os.str();
  }
};

using int2 = intN<2, int8_t>;
using uint2 = intN<2, uint8_t>;
using int4 = intN<4, int8_t>;
using uint4 = intN<4, uint8_t>;

namespace internal {

template <typename intN>
struct intN_numeric_limits_base {
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_integer = true;
  static inline constexpr const bool is_exact = true;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_quiet_NaN = false;
  static inline constexpr const bool has_signaling_NaN = false;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_absent;
  static inline constexpr const bool has_denorm_loss = false;
  static inline constexpr const std::float_round_style round_style =
      std::round_toward_zero;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool is_bounded = true;
  static inline constexpr const int max_digits10 = 0;  // Not used for integers.
  static inline constexpr const int radix = 2;
  static inline constexpr const int min_exponent = 0;
  static inline constexpr const int min_exponent10 = 0;
  static inline constexpr const int max_exponent = 0;
  static inline constexpr const int max_exponent10 = 0;
  static inline constexpr const bool traps = true;
  static inline constexpr const bool tinyness_before = false;
  static inline constexpr const bool is_signed =
      std::is_signed_v<typename intN::underlying_type>;
  static inline constexpr const bool is_modulo = !is_signed;
  static inline constexpr const int digits = intN::digits;
  // floor(digits * log10(2))
  static inline constexpr const int digits10 = (digits * 3) / 10;

  static constexpr intN epsilon() noexcept { return intN(0); }
  static constexpr intN round_error() noexcept { return intN(0); }
  static constexpr intN infinity() noexcept { return intN(0); }
  static constexpr intN quiet_NaN() noexcept { return intN(0); }
  static constexpr intN signaling_NaN() noexcept { return intN(0); }
  static constexpr intN denorm_min() noexcept { return intN(0); }
  static constexpr intN min() noexcept { return intN::lowest(); }
  static constexpr intN lowest() noexcept { return intN::lowest(); }
  static constexpr intN max() noexcept { return intN::highest(); }
};

}  // namespace internal

}  // namespace ml_dtypes

namespace std {

template <>
struct numeric_limits<ml_dtypes::int2>
    : public ml_dtypes::internal::intN_numeric_limits_base<ml_dtypes::int2> {};
template <>
struct numeric_limits<ml_dtypes::uint2>
    : public ml_dtypes::internal::intN_numeric_limits_base<ml_dtypes::uint2> {};
template <>
struct numeric_limits<ml_dtypes::int4>
    : public ml_dtypes::internal::intN_numeric_limits_base<ml_dtypes::int4> {};
template <>
struct numeric_limits<ml_dtypes::uint4>
    : public ml_dtypes::internal::intN_numeric_limits_base<ml_dtypes::uint4> {};

}  // namespace std

#endif  // ML_DTYPES_INTN_H_
