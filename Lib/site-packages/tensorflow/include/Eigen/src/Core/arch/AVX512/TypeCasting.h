// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_AVX512_H
#define EIGEN_TYPE_CASTING_AVX512_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

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

template <>
struct type_casting_traits<double, int64_t> : vectorized_type_casting_traits<double, int64_t> {};
template <>
struct type_casting_traits<int64_t, double> : vectorized_type_casting_traits<int64_t, double> {};

template <>
struct type_casting_traits<half, float> : vectorized_type_casting_traits<half, float> {};
template <>
struct type_casting_traits<float, half> : vectorized_type_casting_traits<float, half> {};

template <>
struct type_casting_traits<bfloat16, float> : vectorized_type_casting_traits<bfloat16, float> {};
template <>
struct type_casting_traits<float, bfloat16> : vectorized_type_casting_traits<float, bfloat16> {};

template <>
EIGEN_STRONG_INLINE Packet16b pcast<Packet16f, Packet16b>(const Packet16f& a) {
  __mmask16 mask = _mm512_cmpneq_ps_mask(a, pzero(a));
  return _mm512_maskz_cvtepi32_epi8(mask, _mm512_set1_epi32(1));
}

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16b, Packet16f>(const Packet16b& a) {
  return _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_cvtepi8_epi32(a), _mm512_set1_epi32(1)));
}

template <>
EIGEN_STRONG_INLINE Packet16i pcast<Packet16f, Packet16i>(const Packet16f& a) {
  return _mm512_cvttps_epi32(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet16f, Packet8d>(const Packet16f& a) {
  return _mm512_cvtps_pd(_mm512_castps512_ps256(a));
}

template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet8f, Packet8d>(const Packet8f& a) {
  return _mm512_cvtps_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet8l pcast<Packet8d, Packet8l>(const Packet8d& a) {
#if defined(EIGEN_VECTORIZE_AVX512DQ) && defined(EIGEN_VECTORIZE_AVX512VL)
  return _mm512_cvttpd_epi64(a);
#else
  constexpr int kTotalBits = sizeof(double) * CHAR_BIT, kMantissaBits = std::numeric_limits<double>::digits - 1,
                kExponentBits = kTotalBits - kMantissaBits - 1, kBias = (1 << (kExponentBits - 1)) - 1;

  const __m512i cst_one = _mm512_set1_epi64(1);
  const __m512i cst_total_bits = _mm512_set1_epi64(kTotalBits);
  const __m512i cst_bias = _mm512_set1_epi64(kBias);

  __m512i a_bits = _mm512_castpd_si512(a);
  // shift left by 1 to clear the sign bit, and shift right by kMantissaBits + 1 to recover biased exponent
  __m512i biased_e = _mm512_srli_epi64(_mm512_slli_epi64(a_bits, 1), kMantissaBits + 1);
  __m512i e = _mm512_sub_epi64(biased_e, cst_bias);

  // shift to the left by kExponentBits + 1 to clear the sign and exponent bits
  __m512i shifted_mantissa = _mm512_slli_epi64(a_bits, kExponentBits + 1);
  // shift to the right by kTotalBits - e to convert the significand to an integer
  __m512i result_significand = _mm512_srlv_epi64(shifted_mantissa, _mm512_sub_epi64(cst_total_bits, e));

  // add the implied bit
  __m512i result_exponent = _mm512_sllv_epi64(cst_one, e);
  // e <= 0 is interpreted as a large positive shift (2's complement), which also conveniently results in zero
  __m512i result = _mm512_add_epi64(result_significand, result_exponent);
  // handle negative arguments
  __mmask8 sign_mask = _mm512_cmplt_epi64_mask(a_bits, _mm512_setzero_si512());
  result = _mm512_mask_sub_epi64(result, sign_mask, _mm512_setzero_si512(), result);
  return result;
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16i, Packet16f>(const Packet16i& a) {
  return _mm512_cvtepi32_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet16i, Packet8d>(const Packet16i& a) {
  return _mm512_cvtepi32_pd(_mm512_castsi512_si256(a));
}

template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet8i, Packet8d>(const Packet8i& a) {
  return _mm512_cvtepi32_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet8l, Packet8d>(const Packet8l& a) {
#if defined(EIGEN_VECTORIZE_AVX512DQ) && defined(EIGEN_VECTORIZE_AVX512VL)
  return _mm512_cvtepi64_pd(a);
#else
  EIGEN_ALIGN64 int64_t aux[8];
  pstore(aux, a);
  return _mm512_set_pd(static_cast<double>(aux[7]), static_cast<double>(aux[6]), static_cast<double>(aux[5]),
                       static_cast<double>(aux[4]), static_cast<double>(aux[3]), static_cast<double>(aux[2]),
                       static_cast<double>(aux[1]), static_cast<double>(aux[0]));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet8d, Packet16f>(const Packet8d& a, const Packet8d& b) {
  return cat256(_mm512_cvtpd_ps(a), _mm512_cvtpd_ps(b));
}

template <>
EIGEN_STRONG_INLINE Packet16i pcast<Packet8d, Packet16i>(const Packet8d& a, const Packet8d& b) {
  return cat256i(_mm512_cvttpd_epi32(a), _mm512_cvttpd_epi32(b));
}

template <>
EIGEN_STRONG_INLINE Packet8i pcast<Packet8d, Packet8i>(const Packet8d& a) {
  return _mm512_cvtpd_epi32(a);
}
template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet8d, Packet8f>(const Packet8d& a) {
  return _mm512_cvtpd_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet16i preinterpret<Packet16i, Packet16f>(const Packet16f& a) {
  return _mm512_castps_si512(a);
}

template <>
EIGEN_STRONG_INLINE Packet16f preinterpret<Packet16f, Packet16i>(const Packet16i& a) {
  return _mm512_castsi512_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d preinterpret<Packet8d, Packet16f>(const Packet16f& a) {
  return _mm512_castps_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d preinterpret<Packet8d, Packet8l>(const Packet8l& a) {
  return _mm512_castsi512_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet8l preinterpret<Packet8l, Packet8d>(const Packet8d& a) {
  return _mm512_castpd_si512(a);
}

template <>
EIGEN_STRONG_INLINE Packet16f preinterpret<Packet16f, Packet8d>(const Packet8d& a) {
  return _mm512_castpd_ps(a);
}

template <>
EIGEN_STRONG_INLINE Packet8f preinterpret<Packet8f, Packet16f>(const Packet16f& a) {
  return _mm512_castps512_ps256(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f preinterpret<Packet4f, Packet16f>(const Packet16f& a) {
  return _mm512_castps512_ps128(a);
}

template <>
EIGEN_STRONG_INLINE Packet4d preinterpret<Packet4d, Packet8d>(const Packet8d& a) {
  return _mm512_castpd512_pd256(a);
}

template <>
EIGEN_STRONG_INLINE Packet2d preinterpret<Packet2d, Packet8d>(const Packet8d& a) {
  return _mm512_castpd512_pd128(a);
}

template <>
EIGEN_STRONG_INLINE Packet16f preinterpret<Packet16f, Packet8f>(const Packet8f& a) {
  return _mm512_castps256_ps512(a);
}

template <>
EIGEN_STRONG_INLINE Packet16f preinterpret<Packet16f, Packet4f>(const Packet4f& a) {
  return _mm512_castps128_ps512(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d preinterpret<Packet8d, Packet4d>(const Packet4d& a) {
  return _mm512_castpd256_pd512(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d preinterpret<Packet8d, Packet2d>(const Packet2d& a) {
  return _mm512_castpd128_pd512(a);
}

template <>
EIGEN_STRONG_INLINE Packet8i preinterpret<Packet8i, Packet16i>(const Packet16i& a) {
  return _mm512_castsi512_si256(a);
}
template <>
EIGEN_STRONG_INLINE Packet4i preinterpret<Packet4i, Packet16i>(const Packet16i& a) {
  return _mm512_castsi512_si128(a);
}

template <>
EIGEN_STRONG_INLINE Packet8h preinterpret<Packet8h, Packet16h>(const Packet16h& a) {
  return _mm256_castsi256_si128(a);
}

template <>
EIGEN_STRONG_INLINE Packet8bf preinterpret<Packet8bf, Packet16bf>(const Packet16bf& a) {
  return _mm256_castsi256_si128(a);
}

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16h, Packet16f>(const Packet16h& a) {
  return half2float(a);
}

template <>
EIGEN_STRONG_INLINE Packet16h pcast<Packet16f, Packet16h>(const Packet16f& a) {
  return float2half(a);
}

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16bf, Packet16f>(const Packet16bf& a) {
  return Bf16ToF32(a);
}

template <>
EIGEN_STRONG_INLINE Packet16bf pcast<Packet16f, Packet16bf>(const Packet16f& a) {
  return F32ToBf16(a);
}

#ifdef EIGEN_VECTORIZE_AVX512FP16

template <>
EIGEN_STRONG_INLINE Packet16h preinterpret<Packet16h, Packet32h>(const Packet32h& a) {
  return _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castph_pd(a), 0));
}
template <>
EIGEN_STRONG_INLINE Packet8h preinterpret<Packet8h, Packet32h>(const Packet32h& a) {
  return _mm256_castsi256_si128(preinterpret<Packet16h>(a));
}

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet32h, Packet16f>(const Packet32h& a) {
  // Discard second-half of input.
  Packet16h low = _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castph_pd(a), 0));
  return _mm512_cvtxph_ps(_mm256_castsi256_ph(low));
}

template <>
EIGEN_STRONG_INLINE Packet32h pcast<Packet16f, Packet32h>(const Packet16f& a, const Packet16f& b) {
  __m512d result = _mm512_undefined_pd();
  result = _mm512_insertf64x4(
      result, _mm256_castsi256_pd(_mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), 0);
  result = _mm512_insertf64x4(
      result, _mm256_castsi256_pd(_mm512_cvtps_ph(b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), 1);
  return _mm512_castpd_ph(result);
}

template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet16h, Packet8f>(const Packet16h& a) {
  // Discard second-half of input.
  Packet8h low = _mm_castps_si128(_mm256_extractf32x4_ps(_mm256_castsi256_ps(a), 0));
  return _mm256_cvtxph_ps(_mm_castsi128_ph(low));
}

template <>
EIGEN_STRONG_INLINE Packet16h pcast<Packet8f, Packet16h>(const Packet8f& a, const Packet8f& b) {
  __m256d result = _mm256_undefined_pd();
  result = _mm256_insertf64x2(result,
                              _mm_castsi128_pd(_mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), 0);
  result = _mm256_insertf64x2(result,
                              _mm_castsi128_pd(_mm256_cvtps_ph(b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), 1);
  return _mm256_castpd_si256(result);
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet8h, Packet4f>(const Packet8h& a) {
  Packet8f full = _mm256_cvtxph_ps(_mm_castsi128_ph(a));
  // Discard second-half of input.
  return _mm256_extractf32x4_ps(full, 0);
}

template <>
EIGEN_STRONG_INLINE Packet8h pcast<Packet4f, Packet8h>(const Packet4f& a, const Packet4f& b) {
  __m256 result = _mm256_undefined_ps();
  result = _mm256_insertf128_ps(result, a, 0);
  result = _mm256_insertf128_ps(result, b, 1);
  return _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
}

#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TYPE_CASTING_AVX512_H
