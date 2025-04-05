// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATHFUNCTIONSIMPL_H
#define EIGEN_MATHFUNCTIONSIMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal Fast reciprocal using Newton-Raphson's method.

 Preconditions:
   1. The starting guess provided in approx_a_recip must have at least half
      the leading mantissa bits in the correct result, such that a single
      Newton-Raphson step is sufficient to get within 1-2 ulps of the currect
      result.
   2. If a is zero, approx_a_recip must be infinite with the same sign as a.
   3. If a is infinite, approx_a_recip must be zero with the same sign as a.

   If the preconditions are satisfied, which they are for for the _*_rcp_ps
   instructions on x86, the result has a maximum relative error of 2 ulps,
   and correctly handles reciprocals of zero, infinity, and NaN.
*/
template <typename Packet, int Steps>
struct generic_reciprocal_newton_step {
  static_assert(Steps > 0, "Steps must be at least 1.");
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& a, const Packet& approx_a_recip) {
    using Scalar = typename unpacket_traits<Packet>::type;
    const Packet two = pset1<Packet>(Scalar(2));
    // Refine the approximation using one Newton-Raphson step:
    //   x_{i} = x_{i-1} * (2 - a * x_{i-1})
    const Packet x = generic_reciprocal_newton_step<Packet, Steps - 1>::run(a, approx_a_recip);
    const Packet tmp = pnmadd(a, x, two);
    // If tmp is NaN, it means that a is either +/-0 or +/-Inf.
    // In this case return the approximation directly.
    const Packet is_not_nan = pcmp_eq(tmp, tmp);
    return pselect(is_not_nan, pmul(x, tmp), x);
  }
};

template <typename Packet>
struct generic_reciprocal_newton_step<Packet, 0> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& /*unused*/, const Packet& approx_rsqrt) {
    return approx_rsqrt;
  }
};

/** \internal Fast reciprocal sqrt using Newton-Raphson's method.

 Preconditions:
   1. The starting guess provided in approx_a_recip must have at least half
      the leading mantissa bits in the correct result, such that a single
      Newton-Raphson step is sufficient to get within 1-2 ulps of the currect
      result.
   2. If a is zero, approx_a_recip must be infinite with the same sign as a.
   3. If a is infinite, approx_a_recip must be zero with the same sign as a.

   If the preconditions are satisfied, which they are for for the _*_rcp_ps
   instructions on x86, the result has a maximum relative error of 2 ulps,
   and correctly handles zero, infinity, and NaN. Positive denormals are
   treated as zero.
*/
template <typename Packet, int Steps>
struct generic_rsqrt_newton_step {
  static_assert(Steps > 0, "Steps must be at least 1.");
  using Scalar = typename unpacket_traits<Packet>::type;
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& a, const Packet& approx_rsqrt) {
    constexpr Scalar kMinusHalf = Scalar(-1) / Scalar(2);
    const Packet cst_minus_half = pset1<Packet>(kMinusHalf);
    const Packet cst_minus_one = pset1<Packet>(Scalar(-1));

    Packet inv_sqrt = approx_rsqrt;
    for (int step = 0; step < Steps; ++step) {
      // Refine the approximation using one Newton-Raphson step:
      // h_n = (x * inv_sqrt) * inv_sqrt - 1 (so that h_n is nearly 0).
      // inv_sqrt = inv_sqrt - 0.5 * inv_sqrt * h_n
      Packet r2 = pmul(a, inv_sqrt);
      Packet half_r = pmul(inv_sqrt, cst_minus_half);
      Packet h_n = pmadd(r2, inv_sqrt, cst_minus_one);
      inv_sqrt = pmadd(half_r, h_n, inv_sqrt);
    }

    // If x is NaN, then either:
    // 1) the input is NaN
    // 2) zero and infinity were multiplied
    // In either of these cases, return approx_rsqrt
    return pselect(pisnan(inv_sqrt), approx_rsqrt, inv_sqrt);
  }
};

template <typename Packet>
struct generic_rsqrt_newton_step<Packet, 0> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& /*unused*/, const Packet& approx_rsqrt) {
    return approx_rsqrt;
  }
};

/** \internal Fast sqrt using Newton-Raphson's method.

 Preconditions:
   1. The starting guess for the reciprocal sqrt provided in approx_rsqrt must
      have at least half the leading mantissa bits in the correct result, such
      that a single Newton-Raphson step is sufficient to get within 1-2 ulps of
      the currect result.
   2. If a is zero, approx_rsqrt must be infinite.
   3. If a is infinite, approx_rsqrt must be zero.

   If the preconditions are satisfied, which they are for for the _*_rsqrt_ps
   instructions on x86, the result has a maximum relative error of 2 ulps,
   and correctly handles zero and infinity, and NaN. Positive denormal inputs
   are treated as zero.
*/
template <typename Packet, int Steps = 1>
struct generic_sqrt_newton_step {
  static_assert(Steps > 0, "Steps must be at least 1.");

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Packet run(const Packet& a, const Packet& approx_rsqrt) {
    using Scalar = typename unpacket_traits<Packet>::type;
    const Packet one_point_five = pset1<Packet>(Scalar(1.5));
    const Packet minus_half = pset1<Packet>(Scalar(-0.5));
    // If a is inf or zero, return a directly.
    const Packet inf_mask = pcmp_eq(a, pset1<Packet>(NumTraits<Scalar>::infinity()));
    const Packet return_a = por(pcmp_eq(a, pzero(a)), inf_mask);
    // Do a single step of Newton's iteration for reciprocal square root:
    //   x_{n+1} = x_n * (1.5 + (-0.5 * x_n) * (a * x_n))).
    // The Newton's step is computed this way to avoid over/under-flows.
    Packet rsqrt = pmul(approx_rsqrt, pmadd(pmul(minus_half, approx_rsqrt), pmul(a, approx_rsqrt), one_point_five));
    for (int step = 1; step < Steps; ++step) {
      rsqrt = pmul(rsqrt, pmadd(pmul(minus_half, rsqrt), pmul(a, rsqrt), one_point_five));
    }

    // Return sqrt(x) = x * rsqrt(x) for non-zero finite positive arguments.
    // Return a itself for 0 or +inf, NaN for negative arguments.
    return pselect(return_a, a, pmul(a, rsqrt));
  }
};

template <typename RealScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RealScalar positive_real_hypot(const RealScalar& x, const RealScalar& y) {
  // IEEE IEC 6059 special cases.
  if ((numext::isinf)(x) || (numext::isinf)(y)) return NumTraits<RealScalar>::infinity();
  if ((numext::isnan)(x) || (numext::isnan)(y)) return NumTraits<RealScalar>::quiet_NaN();

  EIGEN_USING_STD(sqrt);
  RealScalar p, qp;
  p = numext::maxi(x, y);
  if (numext::is_exactly_zero(p)) return RealScalar(0);
  qp = numext::mini(y, x) / p;
  return p * sqrt(RealScalar(1) + qp * qp);
}

template <typename Scalar>
struct hypot_impl {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static EIGEN_DEVICE_FUNC inline RealScalar run(const Scalar& x, const Scalar& y) {
    EIGEN_USING_STD(abs);
    return positive_real_hypot<RealScalar>(abs(x), abs(y));
  }
};

// Generic complex sqrt implementation that correctly handles corner cases
// according to https://en.cppreference.com/w/cpp/numeric/complex/sqrt
template <typename T>
EIGEN_DEVICE_FUNC std::complex<T> complex_sqrt(const std::complex<T>& z) {
  // Computes the principal sqrt of the input.
  //
  // For a complex square root of the number x + i*y. We want to find real
  // numbers u and v such that
  //    (u + i*v)^2 = x + i*y  <=>
  //    u^2 - v^2 + i*2*u*v = x + i*v.
  // By equating the real and imaginary parts we get:
  //    u^2 - v^2 = x
  //    2*u*v = y.
  //
  // For x >= 0, this has the numerically stable solution
  //    u = sqrt(0.5 * (x + sqrt(x^2 + y^2)))
  //    v = y / (2 * u)
  // and for x < 0,
  //    v = sign(y) * sqrt(0.5 * (-x + sqrt(x^2 + y^2)))
  //    u = y / (2 * v)
  //
  // Letting w = sqrt(0.5 * (|x| + |z|)),
  //   if x == 0: u = w, v = sign(y) * w
  //   if x > 0:  u = w, v = y / (2 * w)
  //   if x < 0:  u = |y| / (2 * w), v = sign(y) * w

  const T x = numext::real(z);
  const T y = numext::imag(z);
  const T zero = T(0);
  const T w = numext::sqrt(T(0.5) * (numext::abs(x) + numext::hypot(x, y)));

  return (numext::isinf)(y)           ? std::complex<T>(NumTraits<T>::infinity(), y)
         : numext::is_exactly_zero(x) ? std::complex<T>(w, y < zero ? -w : w)
         : x > zero                   ? std::complex<T>(w, y / (2 * w))
                                      : std::complex<T>(numext::abs(y) / (2 * w), y < zero ? -w : w);
}

// Generic complex rsqrt implementation.
template <typename T>
EIGEN_DEVICE_FUNC std::complex<T> complex_rsqrt(const std::complex<T>& z) {
  // Computes the principal reciprocal sqrt of the input.
  //
  // For a complex reciprocal square root of the number z = x + i*y. We want to
  // find real numbers u and v such that
  //    (u + i*v)^2 = 1 / (x + i*y)  <=>
  //    u^2 - v^2 + i*2*u*v = x/|z|^2 - i*v/|z|^2.
  // By equating the real and imaginary parts we get:
  //    u^2 - v^2 = x/|z|^2
  //    2*u*v = y/|z|^2.
  //
  // For x >= 0, this has the numerically stable solution
  //    u = sqrt(0.5 * (x + |z|)) / |z|
  //    v = -y / (2 * u * |z|)
  // and for x < 0,
  //    v = -sign(y) * sqrt(0.5 * (-x + |z|)) / |z|
  //    u = -y / (2 * v * |z|)
  //
  // Letting w = sqrt(0.5 * (|x| + |z|)),
  //   if x == 0: u = w / |z|, v = -sign(y) * w / |z|
  //   if x > 0:  u = w / |z|, v = -y / (2 * w * |z|)
  //   if x < 0:  u = |y| / (2 * w * |z|), v = -sign(y) * w / |z|

  const T x = numext::real(z);
  const T y = numext::imag(z);
  const T zero = T(0);

  const T abs_z = numext::hypot(x, y);
  const T w = numext::sqrt(T(0.5) * (numext::abs(x) + abs_z));
  const T woz = w / abs_z;
  // Corner cases consistent with 1/sqrt(z) on gcc/clang.
  return numext::is_exactly_zero(abs_z) ? std::complex<T>(NumTraits<T>::infinity(), NumTraits<T>::quiet_NaN())
         : ((numext::isinf)(x) || (numext::isinf)(y)) ? std::complex<T>(zero, zero)
         : numext::is_exactly_zero(x)                 ? std::complex<T>(woz, y < zero ? woz : -woz)
         : x > zero                                   ? std::complex<T>(woz, -y / (2 * w * abs_z))
                    : std::complex<T>(numext::abs(y) / (2 * w * abs_z), y < zero ? woz : -woz);
}

template <typename T>
EIGEN_DEVICE_FUNC std::complex<T> complex_log(const std::complex<T>& z) {
  // Computes complex log.
  T a = numext::abs(z);
  EIGEN_USING_STD(atan2);
  T b = atan2(z.imag(), z.real());
  return std::complex<T>(numext::log(a), b);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATHFUNCTIONSIMPL_H
