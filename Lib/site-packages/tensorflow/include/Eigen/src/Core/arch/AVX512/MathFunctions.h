// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Pedro Gonnet (pedro.gonnet@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
#define THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet16f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet8d)

template <>
EIGEN_STRONG_INLINE Packet16h pfrexp(const Packet16h& a, Packet16h& exponent) {
  Packet16f fexponent;
  const Packet16h out = float2half(pfrexp<Packet16f>(half2float(a), fexponent));
  exponent = float2half(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet16h pldexp(const Packet16h& a, const Packet16h& exponent) {
  return float2half(pldexp<Packet16f>(half2float(a), half2float(exponent)));
}

template <>
EIGEN_STRONG_INLINE Packet16bf pfrexp(const Packet16bf& a, Packet16bf& exponent) {
  Packet16f fexponent;
  const Packet16bf out = F32ToBf16(pfrexp<Packet16f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet16bf pldexp(const Packet16bf& a, const Packet16bf& exponent) {
  return F32ToBf16(pldexp<Packet16f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

#if EIGEN_FAST_MATH
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16f psqrt<Packet16f>(const Packet16f& _x) {
  return generic_sqrt_newton_step<Packet16f>::run(_x, _mm512_rsqrt14_ps(_x));
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet8d psqrt<Packet8d>(const Packet8d& _x) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return generic_sqrt_newton_step<Packet8d, /*Steps=*/1>::run(_x, _mm512_rsqrt28_pd(_x));
#else
  return generic_sqrt_newton_step<Packet8d, /*Steps=*/2>::run(_x, _mm512_rsqrt14_pd(_x));
#endif
}
#else
template <>
EIGEN_STRONG_INLINE Packet16f psqrt<Packet16f>(const Packet16f& x) {
  return _mm512_sqrt_ps(x);
}

template <>
EIGEN_STRONG_INLINE Packet8d psqrt<Packet8d>(const Packet8d& x) {
  return _mm512_sqrt_pd(x);
}
#endif

// prsqrt for float.
#if defined(EIGEN_VECTORIZE_AVX512ER)
template <>
EIGEN_STRONG_INLINE Packet16f prsqrt<Packet16f>(const Packet16f& x) {
  return _mm512_rsqrt28_ps(x);
}
#elif EIGEN_FAST_MATH

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16f prsqrt<Packet16f>(const Packet16f& _x) {
  return generic_rsqrt_newton_step<Packet16f, /*Steps=*/1>::run(_x, _mm512_rsqrt14_ps(_x));
}
#endif

// prsqrt for double.
#if EIGEN_FAST_MATH
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet8d prsqrt<Packet8d>(const Packet8d& _x) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return generic_rsqrt_newton_step<Packet8d, /*Steps=*/1>::run(_x, _mm512_rsqrt28_pd(_x));
#else
  return generic_rsqrt_newton_step<Packet8d, /*Steps=*/2>::run(_x, _mm512_rsqrt14_pd(_x));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f preciprocal<Packet16f>(const Packet16f& a) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return _mm512_rcp28_ps(a);
#else
  return generic_reciprocal_newton_step<Packet16f, /*Steps=*/1>::run(a, _mm512_rcp14_ps(a));
#endif
}
#endif

BF16_PACKET_FUNCTION(Packet16f, Packet16bf, pcos)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, pexp)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, pexpm1)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, plog)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, plog1p)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, plog2)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, preciprocal)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, prsqrt)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, psin)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, psqrt)
BF16_PACKET_FUNCTION(Packet16f, Packet16bf, ptanh)
F16_PACKET_FUNCTION(Packet16f, Packet16h, pcos)
F16_PACKET_FUNCTION(Packet16f, Packet16h, pexp)
F16_PACKET_FUNCTION(Packet16f, Packet16h, pexpm1)
F16_PACKET_FUNCTION(Packet16f, Packet16h, plog)
F16_PACKET_FUNCTION(Packet16f, Packet16h, plog1p)
F16_PACKET_FUNCTION(Packet16f, Packet16h, plog2)
F16_PACKET_FUNCTION(Packet16f, Packet16h, preciprocal)
F16_PACKET_FUNCTION(Packet16f, Packet16h, prsqrt)
F16_PACKET_FUNCTION(Packet16f, Packet16h, psin)
F16_PACKET_FUNCTION(Packet16f, Packet16h, psqrt)
F16_PACKET_FUNCTION(Packet16f, Packet16h, ptanh)

}  // end namespace internal

}  // end namespace Eigen

#endif  // THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
