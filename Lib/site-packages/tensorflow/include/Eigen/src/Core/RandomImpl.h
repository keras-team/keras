// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charles Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RANDOM_IMPL_H
#define EIGEN_RANDOM_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/****************************************************************************
 * Implementation of random                                               *
 ****************************************************************************/

template <typename Scalar, bool IsComplex, bool IsInteger>
struct random_default_impl {};

template <typename Scalar>
struct random_impl : random_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template <typename Scalar>
struct random_retval {
  typedef Scalar type;
};

template <typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y) {
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run(x, y);
}

template <typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random() {
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run();
}

// TODO: replace or provide alternatives to this, e.g. std::random_device
struct eigen_random_device {
  using ReturnType = int;
  static constexpr int Entropy = meta_floor_log2<(unsigned int)(RAND_MAX) + 1>::value;
  static constexpr ReturnType Highest = RAND_MAX;
  static EIGEN_DEVICE_FUNC inline ReturnType run() { return std::rand(); }
};

// Fill a built-in unsigned integer with numRandomBits beginning with the least significant bit
template <typename Scalar>
struct random_bits_impl {
  EIGEN_STATIC_ASSERT(std::is_unsigned<Scalar>::value, SCALAR MUST BE A BUILT - IN UNSIGNED INTEGER)
  using RandomDevice = eigen_random_device;
  using RandomReturnType = typename RandomDevice::ReturnType;
  static constexpr int kEntropy = RandomDevice::Entropy;
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  // return a Scalar filled with numRandomBits beginning from the least significant bit
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert((numRandomBits >= 0) && (numRandomBits <= kTotalBits));
    const Scalar mask = Scalar(-1) >> ((kTotalBits - numRandomBits) & (kTotalBits - 1));
    Scalar randomBits = 0;
    for (int shift = 0; shift < numRandomBits; shift += kEntropy) {
      RandomReturnType r = RandomDevice::run();
      randomBits |= static_cast<Scalar>(r) << shift;
    }
    // clear the excess bits
    randomBits &= mask;
    return randomBits;
  }
};

template <typename BitsType>
EIGEN_DEVICE_FUNC inline BitsType getRandomBits(int numRandomBits) {
  return random_bits_impl<BitsType>::run(numRandomBits);
}

// random implementation for a built-in floating point type
template <typename Scalar, bool BuiltIn = std::is_floating_point<Scalar>::value>
struct random_float_impl {
  using BitsType = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() {
    const int digits = NumTraits<Scalar>::digits();
    return digits - 1;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    BitsType randomBits = getRandomBits<BitsType>(numRandomBits);
    // if fewer than MantissaBits is requested, shift them to the left
    randomBits <<= (mantissaBits() - numRandomBits);
    // randomBits is in the half-open interval [2,4)
    randomBits |= numext::bit_cast<BitsType>(Scalar(2));
    // result is in the half-open interval [-1,1)
    Scalar result = numext::bit_cast<Scalar>(randomBits) - Scalar(3);
    return result;
  }
};
// random implementation for a custom floating point type
// uses double as the implementation with a mantissa with a size equal to either the target scalar's mantissa or that of
// double, whichever is smaller
template <typename Scalar>
struct random_float_impl<Scalar, false> {
  static EIGEN_DEVICE_FUNC inline int mantissaBits() {
    const int digits = NumTraits<Scalar>::digits();
    constexpr int kDoubleDigits = NumTraits<double>::digits();
    return numext::mini(digits, kDoubleDigits) - 1;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    Scalar result = static_cast<Scalar>(random_float_impl<double>::run(numRandomBits));
    return result;
  }
};

// random implementation for long double
// this specialization is not compatible with double-double scalars
template <bool Specialize = (sizeof(long double) == 2 * sizeof(uint64_t)) &&
                            ((std::numeric_limits<long double>::digits != (2 * std::numeric_limits<double>::digits)))>
struct random_longdouble_impl {
  static constexpr int Size = sizeof(long double);
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() { return NumTraits<long double>::digits() - 1; }
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    EIGEN_USING_STD(memcpy);
    int numLowBits = numext::mini(numRandomBits, 64);
    int numHighBits = numext::maxi(numRandomBits - 64, 0);
    uint64_t randomBits[2];
    long double result = 2.0L;
    memcpy(&randomBits, &result, Size);
    randomBits[0] |= getRandomBits<uint64_t>(numLowBits);
    randomBits[1] |= getRandomBits<uint64_t>(numHighBits);
    memcpy(&result, &randomBits, Size);
    result -= 3.0L;
    return result;
  }
};
template <>
struct random_longdouble_impl<false> {
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() { return NumTraits<double>::digits() - 1; }
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits) {
    return static_cast<long double>(random_float_impl<double>::run(numRandomBits));
  }
};
template <>
struct random_float_impl<long double> : random_longdouble_impl<> {};

template <typename Scalar>
struct random_default_impl<Scalar, false, false> {
  using Impl = random_float_impl<Scalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    Scalar half_x = Scalar(0.5) * x;
    Scalar half_y = Scalar(0.5) * y;
    Scalar result = (half_x + half_y) + (half_y - half_x) * run(numRandomBits);
    // result is in the half-open interval [x, y) -- provided that x < y
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return run(x, y, Impl::mantissaBits());
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) { return Impl::run(numRandomBits); }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return run(Impl::mantissaBits()); }
};

template <typename Scalar, bool IsSigned = NumTraits<Scalar>::IsSigned, bool BuiltIn = std::is_integral<Scalar>::value>
struct random_int_impl;

// random implementation for a built-in unsigned integer type
template <typename Scalar>
struct random_int_impl<Scalar, false, true> {
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    Scalar range = y - x;
    // handle edge case where [x,y] spans the entire range of Scalar
    if (range == NumTraits<Scalar>::highest()) return run();
    Scalar count = range + 1;
    // calculate the number of random bits needed to fill range
    int numRandomBits = log2_ceil(count);
    Scalar randomBits;
    do {
      randomBits = getRandomBits<Scalar>(numRandomBits);
      // if the random draw is outside [0, range), try again (rejection sampling)
      // in the worst-case scenario, the probability of rejection is: 1/2 - 1/2^numRandomBits < 50%
    } while (randomBits >= count);
    Scalar result = x + randomBits;
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return getRandomBits<Scalar>(kTotalBits); }
};

// random implementation for a built-in signed integer type
template <typename Scalar>
struct random_int_impl<Scalar, true, true> {
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  using BitsType = typename make_unsigned<Scalar>::type;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    // Avoid overflow by representing `range` as an unsigned type
    BitsType range = static_cast<BitsType>(y) - static_cast<BitsType>(x);
    BitsType randomBits = random_int_impl<BitsType>::run(0, range);
    // Avoid overflow in the case where `x` is negative and there is a large range so
    // `randomBits` would also be negative if cast to `Scalar` first.
    Scalar result = static_cast<Scalar>(static_cast<BitsType>(x) + randomBits);
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return static_cast<Scalar>(getRandomBits<BitsType>(kTotalBits)); }
};

// todo: custom integers
template <typename Scalar, bool IsSigned>
struct random_int_impl<Scalar, IsSigned, false> {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar&, const Scalar&) { return run(); }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
    eigen_assert(std::false_type::value && "RANDOM FOR CUSTOM INTEGERS NOT YET SUPPORTED");
    return Scalar(0);
  }
};

template <typename Scalar>
struct random_default_impl<Scalar, false, true> : random_int_impl<Scalar> {};

template <>
struct random_impl<bool> {
  static EIGEN_DEVICE_FUNC inline bool run(const bool& x, const bool& y) {
    if (y <= x) return x;
    return run();
  }
  static EIGEN_DEVICE_FUNC inline bool run() { return getRandomBits<unsigned>(1) ? true : false; }
};

template <typename Scalar>
struct random_default_impl<Scalar, true, false> {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  using Impl = random_impl<RealScalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    return Scalar(Impl::run(x.real(), y.real(), numRandomBits), Impl::run(x.imag(), y.imag(), numRandomBits));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return Scalar(Impl::run(x.real(), y.real()), Impl::run(x.imag(), y.imag()));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    return Scalar(Impl::run(numRandomBits), Impl::run(numRandomBits));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return Scalar(Impl::run(), Impl::run()); }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_RANDOM_IMPL_H
