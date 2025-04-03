// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_SSE_H
#define EIGEN_TYPE_CASTING_SSE_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#ifndef EIGEN_VECTORIZE_AVX
template <>
struct type_casting_traits<float, bool> : vectorized_type_casting_traits<float, bool> {};
template <>
struct type_casting_traits<bool, float> : vectorized_type_casting_traits<bool, float> {};

template <>
struct type_casting_traits<float, int> : vectorized_type_casting_traits<float, int> {};
template <>
struct type_casting_traits<int, float> : vectorized_type_casting_traits<int, float> {};

template <>
struct type_casting_traits<float, double> : vectorized_type_casting_traits<float, double> {};
template <>
struct type_casting_traits<double, float> : vectorized_type_casting_traits<double, float> {};

template <>
struct type_casting_traits<double, int> : vectorized_type_casting_traits<double, int> {};
template <>
struct type_casting_traits<int, double> : vectorized_type_casting_traits<int, double> {};

#ifndef EIGEN_VECTORIZE_AVX2
template <>
struct type_casting_traits<double, int64_t> : vectorized_type_casting_traits<double, int64_t> {};
template <>
struct type_casting_traits<int64_t, double> : vectorized_type_casting_traits<int64_t, double> {};
#endif
#endif

template <>
EIGEN_STRONG_INLINE Packet16b pcast<Packet4f, Packet16b>(const Packet4f& a, const Packet4f& b, const Packet4f& c,
                                                         const Packet4f& d) {
  __m128 zero = pzero(a);
  __m128 nonzero_a = _mm_cmpneq_ps(a, zero);
  __m128 nonzero_b = _mm_cmpneq_ps(b, zero);
  __m128 nonzero_c = _mm_cmpneq_ps(c, zero);
  __m128 nonzero_d = _mm_cmpneq_ps(d, zero);
  __m128i ab_bytes = _mm_packs_epi32(_mm_castps_si128(nonzero_a), _mm_castps_si128(nonzero_b));
  __m128i cd_bytes = _mm_packs_epi32(_mm_castps_si128(nonzero_c), _mm_castps_si128(nonzero_d));
  __m128i merged = _mm_packs_epi16(ab_bytes, cd_bytes);
  return _mm_and_si128(merged, _mm_set1_epi8(1));
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet16b, Packet4f>(const Packet16b& a) {
  const __m128 cst_one = _mm_set_ps1(1.0f);
#ifdef EIGEN_VECTORIZE_SSE4_1
  __m128i a_extended = _mm_cvtepi8_epi32(a);
  __m128i abcd = _mm_cmpeq_epi32(a_extended, _mm_setzero_si128());
#else
  __m128i abcd_efhg_ijkl_mnop = _mm_cmpeq_epi8(a, _mm_setzero_si128());
  __m128i aabb_ccdd_eeff_gghh = _mm_unpacklo_epi8(abcd_efhg_ijkl_mnop, abcd_efhg_ijkl_mnop);
  __m128i abcd = _mm_unpacklo_epi8(aabb_ccdd_eeff_gghh, aabb_ccdd_eeff_gghh);
#endif
  __m128 result = _mm_andnot_ps(_mm_castsi128_ps(abcd), cst_one);
  return result;
}

template <>
EIGEN_STRONG_INLINE Packet4i pcast<Packet4f, Packet4i>(const Packet4f& a) {
  return _mm_cvttps_epi32(a);
}

template <>
EIGEN_STRONG_INLINE Packet4i pcast<Packet2d, Packet4i>(const Packet2d& a, const Packet2d& b) {
  return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_cvttpd_epi32(a)), _mm_castsi128_ps(_mm_cvttpd_epi32(b)),
                                         (1 << 2) | (1 << 6)));
}

template <>
EIGEN_STRONG_INLINE Packet2l pcast<Packet2d, Packet2l>(const Packet2d& a) {
#if EIGEN_ARCH_x86_64
  return _mm_set_epi64x(_mm_cvttsd_si64(preverse(a)), _mm_cvttsd_si64(a));
#else
  return _mm_set_epi64x(static_cast<int64_t>(pfirst(preverse(a))), static_cast<int64_t>(pfirst(a)));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet2d pcast<Packet2l, Packet2d>(const Packet2l& a) {
  EIGEN_ALIGN16 int64_t aux[2];
  pstore(aux, a);
  return _mm_set_pd(static_cast<double>(aux[1]), static_cast<double>(aux[0]));
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet4i, Packet4f>(const Packet4i& a) {
  return _mm_cvtepi32_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet2d, Packet4f>(const Packet2d& a, const Packet2d& b) {
  return _mm_shuffle_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b), (1 << 2) | (1 << 6));
}

template <>
EIGEN_STRONG_INLINE Packet2d pcast<Packet4i, Packet2d>(const Packet4i& a) {
  // Simply discard the second half of the input
  return _mm_cvtepi32_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet2d pcast<Packet4f, Packet2d>(const Packet4f& a) {
  // Simply discard the second half of the input
  return _mm_cvtps_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet2d preinterpret<Packet2d, Packet4f>(const Packet4f& a) {
  return _mm_castps_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f preinterpret<Packet4f, Packet2d>(const Packet2d& a) {
  return _mm_castpd_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet4i preinterpret<Packet4i, Packet4f>(const Packet4f& a) {
  return _mm_castps_si128(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f preinterpret<Packet4f, Packet4i>(const Packet4i& a) {
  return _mm_castsi128_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet2d preinterpret<Packet2d, Packet4i>(const Packet4i& a) {
  return _mm_castsi128_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet2d preinterpret<Packet2d, Packet2l>(const Packet2l& a) {
  return _mm_castsi128_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet2l preinterpret<Packet2l, Packet2d>(const Packet2d& a) {
  return _mm_castpd_si128(a);
}

template <>
EIGEN_STRONG_INLINE Packet4i preinterpret<Packet4i, Packet2d>(const Packet2d& a) {
  return _mm_castpd_si128(a);
}

template <>
EIGEN_STRONG_INLINE Packet4ui preinterpret<Packet4ui, Packet4i>(const Packet4i& a) {
  return Packet4ui(a);
}

template <>
EIGEN_STRONG_INLINE Packet4i preinterpret<Packet4i, Packet4ui>(const Packet4ui& a) {
  return Packet4i(a);
}

// Disable the following code since it's broken on too many platforms / compilers.
// #elif defined(EIGEN_VECTORIZE_SSE) && (!EIGEN_ARCH_x86_64) && (!EIGEN_COMP_MSVC)
#if 0

template <>
struct type_casting_traits<Eigen::half, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet4h, Packet4f>(const Packet4h& a) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  Eigen::half h = raw_uint16_to_half(static_cast<unsigned short>(a64));
  float f1 = static_cast<float>(h);
  h = raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  float f2 = static_cast<float>(h);
  h = raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  float f3 = static_cast<float>(h);
  h = raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  float f4 = static_cast<float>(h);
  return _mm_set_ps(f4, f3, f2, f1);
}

template <>
struct type_casting_traits<float, Eigen::half> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_STRONG_INLINE Packet4h pcast<Packet4f, Packet4h>(const Packet4f& a) {
  EIGEN_ALIGN16 float aux[4];
  pstore(aux, a);
  Eigen::half h0(aux[0]);
  Eigen::half h1(aux[1]);
  Eigen::half h2(aux[2]);
  Eigen::half h3(aux[3]);

  Packet4h result;
  result.x = _mm_set_pi16(h3.x, h2.x, h1.x, h0.x);
  return result;
}

#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TYPE_CASTING_SSE_H
