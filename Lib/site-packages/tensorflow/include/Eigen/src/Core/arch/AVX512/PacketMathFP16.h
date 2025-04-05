// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
//
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_FP16_AVX512_H
#define EIGEN_PACKET_MATH_FP16_AVX512_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

typedef __m512h Packet32h;
typedef eigen_packet_wrapper<__m256i, 1> Packet16h;
typedef eigen_packet_wrapper<__m128i, 2> Packet8h;

template <>
struct is_arithmetic<Packet8h> {
  enum { value = true };
};

template <>
struct packet_traits<half> : default_packet_traits {
  typedef Packet32h type;
  typedef Packet16h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 32,

    HasCmp = 1,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasLog = 1,
    HasLog1p = 1,
    HasExp = 1,
    HasExpm1 = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    // These ones should be implemented in future
    HasBessel = 0,
    HasNdtri = 0,
    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = 0,  // EIGEN_FAST_MATH,
    HasBlend = 0
  };
};

template <>
struct unpacket_traits<Packet32h> {
  typedef Eigen::half type;
  typedef Packet16h half;
  enum {
    size = 32,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<Packet16h> {
  typedef Eigen::half type;
  typedef Packet8h half;
  enum {
    size = 16,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<Packet8h> {
  typedef Eigen::half type;
  typedef Packet8h half;
  enum {
    size = 8,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

// Memory functions

// pset1

template <>
EIGEN_STRONG_INLINE Packet32h pset1<Packet32h>(const Eigen::half& from) {
  return _mm512_set1_ph(static_cast<_Float16>(from));
}

// pset1frombits
template <>
EIGEN_STRONG_INLINE Packet32h pset1frombits<Packet32h>(unsigned short from) {
  return _mm512_castsi512_ph(_mm512_set1_epi16(from));
}

// pfirst

template <>
EIGEN_STRONG_INLINE Eigen::half pfirst<Packet32h>(const Packet32h& from) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  return half_impl::raw_uint16_to_half(
      static_cast<unsigned short>(_mm256_extract_epi16(_mm512_extracti32x8_epi32(_mm512_castph_si512(from), 0), 0)));
#else
  Eigen::half dest[32];
  _mm512_storeu_ph(dest, from);
  return dest[0];
#endif
}

// pload

template <>
EIGEN_STRONG_INLINE Packet32h pload<Packet32h>(const Eigen::half* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_ph(from);
}

// ploadu

template <>
EIGEN_STRONG_INLINE Packet32h ploadu<Packet32h>(const Eigen::half* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_ph(from);
}

// pstore

template <>
EIGEN_STRONG_INLINE void pstore<half>(Eigen::half* to, const Packet32h& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_ph(to, from);
}

// pstoreu

template <>
EIGEN_STRONG_INLINE void pstoreu<half>(Eigen::half* to, const Packet32h& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_ph(to, from);
}

// ploaddup
template <>
EIGEN_STRONG_INLINE Packet32h ploaddup<Packet32h>(const Eigen::half* from) {
  __m512h a = _mm512_castph256_ph512(_mm256_loadu_ph(from));
  return _mm512_permutexvar_ph(_mm512_set_epi16(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6,
                                                5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0),
                               a);
}

// ploadquad
template <>
EIGEN_STRONG_INLINE Packet32h ploadquad<Packet32h>(const Eigen::half* from) {
  __m512h a = _mm512_castph128_ph512(_mm_loadu_ph(from));
  return _mm512_permutexvar_ph(
      _mm512_set_epi16(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0),
      a);
}

// pabs

template <>
EIGEN_STRONG_INLINE Packet32h pabs<Packet32h>(const Packet32h& a) {
  return _mm512_abs_ph(a);
}

// psignbit

template <>
EIGEN_STRONG_INLINE Packet32h psignbit<Packet32h>(const Packet32h& a) {
  return _mm512_castsi512_ph(_mm512_srai_epi16(_mm512_castph_si512(a), 15));
}

// pmin

template <>
EIGEN_STRONG_INLINE Packet32h pmin<Packet32h>(const Packet32h& a, const Packet32h& b) {
  return _mm512_min_ph(a, b);
}

// pmax

template <>
EIGEN_STRONG_INLINE Packet32h pmax<Packet32h>(const Packet32h& a, const Packet32h& b) {
  return _mm512_max_ph(a, b);
}

// plset
template <>
EIGEN_STRONG_INLINE Packet32h plset<Packet32h>(const half& a) {
  return _mm512_add_ph(_mm512_set1_ph(a),
                       _mm512_set_ph(31.0f, 30.0f, 29.0f, 28.0f, 27.0f, 26.0f, 25.0f, 24.0f, 23.0f, 22.0f, 21.0f, 20.0f,
                                     19.0f, 18.0f, 17.0f, 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
                                     7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
}

// por

template <>
EIGEN_STRONG_INLINE Packet32h por(const Packet32h& a, const Packet32h& b) {
  return _mm512_castsi512_ph(_mm512_or_si512(_mm512_castph_si512(a), _mm512_castph_si512(b)));
}

// pxor

template <>
EIGEN_STRONG_INLINE Packet32h pxor(const Packet32h& a, const Packet32h& b) {
  return _mm512_castsi512_ph(_mm512_xor_si512(_mm512_castph_si512(a), _mm512_castph_si512(b)));
}

// pand

template <>
EIGEN_STRONG_INLINE Packet32h pand(const Packet32h& a, const Packet32h& b) {
  return _mm512_castsi512_ph(_mm512_and_si512(_mm512_castph_si512(a), _mm512_castph_si512(b)));
}

// pandnot

template <>
EIGEN_STRONG_INLINE Packet32h pandnot(const Packet32h& a, const Packet32h& b) {
  return _mm512_castsi512_ph(_mm512_andnot_si512(_mm512_castph_si512(b), _mm512_castph_si512(a)));
}

// pselect

template <>
EIGEN_DEVICE_FUNC inline Packet32h pselect(const Packet32h& mask, const Packet32h& a, const Packet32h& b) {
  __mmask32 mask32 = _mm512_cmp_epi16_mask(_mm512_castph_si512(mask), _mm512_setzero_epi32(), _MM_CMPINT_EQ);
  return _mm512_mask_blend_ph(mask32, a, b);
}

// pcmp_eq

template <>
EIGEN_STRONG_INLINE Packet32h pcmp_eq(const Packet32h& a, const Packet32h& b) {
  __mmask32 mask = _mm512_cmp_ph_mask(a, b, _CMP_EQ_OQ);
  return _mm512_castsi512_ph(_mm512_mask_set1_epi16(_mm512_set1_epi32(0), mask, 0xffffu));
}

// pcmp_le

template <>
EIGEN_STRONG_INLINE Packet32h pcmp_le(const Packet32h& a, const Packet32h& b) {
  __mmask32 mask = _mm512_cmp_ph_mask(a, b, _CMP_LE_OQ);
  return _mm512_castsi512_ph(_mm512_mask_set1_epi16(_mm512_set1_epi32(0), mask, 0xffffu));
}

// pcmp_lt

template <>
EIGEN_STRONG_INLINE Packet32h pcmp_lt(const Packet32h& a, const Packet32h& b) {
  __mmask32 mask = _mm512_cmp_ph_mask(a, b, _CMP_LT_OQ);
  return _mm512_castsi512_ph(_mm512_mask_set1_epi16(_mm512_set1_epi32(0), mask, 0xffffu));
}

// pcmp_lt_or_nan

template <>
EIGEN_STRONG_INLINE Packet32h pcmp_lt_or_nan(const Packet32h& a, const Packet32h& b) {
  __mmask32 mask = _mm512_cmp_ph_mask(a, b, _CMP_NGE_UQ);
  return _mm512_castsi512_ph(_mm512_mask_set1_epi16(_mm512_set1_epi16(0), mask, 0xffffu));
}

// padd

template <>
EIGEN_STRONG_INLINE Packet32h padd<Packet32h>(const Packet32h& a, const Packet32h& b) {
  return _mm512_add_ph(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16h padd<Packet16h>(const Packet16h& a, const Packet16h& b) {
  return _mm256_castph_si256(_mm256_add_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8h padd<Packet8h>(const Packet8h& a, const Packet8h& b) {
  return _mm_castph_si128(_mm_add_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b)));
}

// psub

template <>
EIGEN_STRONG_INLINE Packet32h psub<Packet32h>(const Packet32h& a, const Packet32h& b) {
  return _mm512_sub_ph(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16h psub<Packet16h>(const Packet16h& a, const Packet16h& b) {
  return _mm256_castph_si256(_mm256_sub_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8h psub<Packet8h>(const Packet8h& a, const Packet8h& b) {
  return _mm_castph_si128(_mm_sub_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b)));
}

// pmul

template <>
EIGEN_STRONG_INLINE Packet32h pmul<Packet32h>(const Packet32h& a, const Packet32h& b) {
  return _mm512_mul_ph(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16h pmul<Packet16h>(const Packet16h& a, const Packet16h& b) {
  return _mm256_castph_si256(_mm256_mul_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8h pmul<Packet8h>(const Packet8h& a, const Packet8h& b) {
  return _mm_castph_si128(_mm_mul_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b)));
}

// pdiv

template <>
EIGEN_STRONG_INLINE Packet32h pdiv<Packet32h>(const Packet32h& a, const Packet32h& b) {
  return _mm512_div_ph(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16h pdiv<Packet16h>(const Packet16h& a, const Packet16h& b) {
  return _mm256_castph_si256(_mm256_div_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b)));
}

template <>
EIGEN_STRONG_INLINE Packet8h pdiv<Packet8h>(const Packet8h& a, const Packet8h& b) {
  return _mm_castph_si128(_mm_div_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b)));
}

// pround

template <>
EIGEN_STRONG_INLINE Packet32h pround<Packet32h>(const Packet32h& a) {
  // Work-around for default std::round rounding mode.

  // Mask for the sign bit
  const Packet32h signMask = pset1frombits<Packet32h>(static_cast<numext::uint16_t>(0x8000u));
  // The largest half-preicision float less than 0.5
  const Packet32h prev0dot5 = pset1frombits<Packet32h>(static_cast<numext::uint16_t>(0x37FFu));

  return _mm512_roundscale_ph(padd(por(pand(a, signMask), prev0dot5), a), _MM_FROUND_TO_ZERO);
}

// print

template <>
EIGEN_STRONG_INLINE Packet32h print<Packet32h>(const Packet32h& a) {
  return _mm512_roundscale_ph(a, _MM_FROUND_CUR_DIRECTION);
}

// pceil

template <>
EIGEN_STRONG_INLINE Packet32h pceil<Packet32h>(const Packet32h& a) {
  return _mm512_roundscale_ph(a, _MM_FROUND_TO_POS_INF);
}

// pfloor

template <>
EIGEN_STRONG_INLINE Packet32h pfloor<Packet32h>(const Packet32h& a) {
  return _mm512_roundscale_ph(a, _MM_FROUND_TO_NEG_INF);
}

// ptrunc

template <>
EIGEN_STRONG_INLINE Packet32h ptrunc<Packet32h>(const Packet32h& a) {
  return _mm512_roundscale_ph(a, _MM_FROUND_TO_ZERO);
}

// predux
template <>
EIGEN_STRONG_INLINE half predux<Packet32h>(const Packet32h& a) {
  return (half)_mm512_reduce_add_ph(a);
}

template <>
EIGEN_STRONG_INLINE half predux<Packet16h>(const Packet16h& a) {
  return (half)_mm256_reduce_add_ph(_mm256_castsi256_ph(a));
}

template <>
EIGEN_STRONG_INLINE half predux<Packet8h>(const Packet8h& a) {
  return (half)_mm_reduce_add_ph(_mm_castsi128_ph(a));
}

// predux_half_dowto4
template <>
EIGEN_STRONG_INLINE Packet16h predux_half_dowto4<Packet32h>(const Packet32h& a) {
#ifdef EIGEN_VECTORIZE_AVX512DQ
  __m256i lowHalf = _mm256_castps_si256(_mm512_extractf32x8_ps(_mm512_castph_ps(a), 0));
  __m256i highHalf = _mm256_castps_si256(_mm512_extractf32x8_ps(_mm512_castph_ps(a), 1));

  return Packet16h(padd<Packet16h>(lowHalf, highHalf));
#else
  Eigen::half data[32];
  _mm512_storeu_ph(data, a);

  __m256i lowHalf = _mm256_castph_si256(_mm256_loadu_ph(data));
  __m256i highHalf = _mm256_castph_si256(_mm256_loadu_ph(data + 16));

  return Packet16h(padd<Packet16h>(lowHalf, highHalf));
#endif
}

// predux_max

// predux_min

// predux_mul

#ifdef EIGEN_VECTORIZE_FMA

// pmadd

template <>
EIGEN_STRONG_INLINE Packet32h pmadd(const Packet32h& a, const Packet32h& b, const Packet32h& c) {
  return _mm512_fmadd_ph(a, b, c);
}

template <>
EIGEN_STRONG_INLINE Packet16h pmadd(const Packet16h& a, const Packet16h& b, const Packet16h& c) {
  return _mm256_castph_si256(_mm256_fmadd_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b), _mm256_castsi256_ph(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8h pmadd(const Packet8h& a, const Packet8h& b, const Packet8h& c) {
  return _mm_castph_si128(_mm_fmadd_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b), _mm_castsi128_ph(c)));
}

// pmsub

template <>
EIGEN_STRONG_INLINE Packet32h pmsub(const Packet32h& a, const Packet32h& b, const Packet32h& c) {
  return _mm512_fmsub_ph(a, b, c);
}

template <>
EIGEN_STRONG_INLINE Packet16h pmsub(const Packet16h& a, const Packet16h& b, const Packet16h& c) {
  return _mm256_castph_si256(_mm256_fmsub_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b), _mm256_castsi256_ph(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8h pmsub(const Packet8h& a, const Packet8h& b, const Packet8h& c) {
  return _mm_castph_si128(_mm_fmsub_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b), _mm_castsi128_ph(c)));
}

// pnmadd

template <>
EIGEN_STRONG_INLINE Packet32h pnmadd(const Packet32h& a, const Packet32h& b, const Packet32h& c) {
  return _mm512_fnmadd_ph(a, b, c);
}

template <>
EIGEN_STRONG_INLINE Packet16h pnmadd(const Packet16h& a, const Packet16h& b, const Packet16h& c) {
  return _mm256_castph_si256(_mm256_fnmadd_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b), _mm256_castsi256_ph(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8h pnmadd(const Packet8h& a, const Packet8h& b, const Packet8h& c) {
  return _mm_castph_si128(_mm_fnmadd_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b), _mm_castsi128_ph(c)));
}

// pnmsub

template <>
EIGEN_STRONG_INLINE Packet32h pnmsub(const Packet32h& a, const Packet32h& b, const Packet32h& c) {
  return _mm512_fnmsub_ph(a, b, c);
}

template <>
EIGEN_STRONG_INLINE Packet16h pnmsub(const Packet16h& a, const Packet16h& b, const Packet16h& c) {
  return _mm256_castph_si256(_mm256_fnmsub_ph(_mm256_castsi256_ph(a), _mm256_castsi256_ph(b), _mm256_castsi256_ph(c)));
}

template <>
EIGEN_STRONG_INLINE Packet8h pnmsub(const Packet8h& a, const Packet8h& b, const Packet8h& c) {
  return _mm_castph_si128(_mm_fnmsub_ph(_mm_castsi128_ph(a), _mm_castsi128_ph(b), _mm_castsi128_ph(c)));
}

#endif

// pnegate

template <>
EIGEN_STRONG_INLINE Packet32h pnegate<Packet32h>(const Packet32h& a) {
  return _mm512_sub_ph(_mm512_set1_ph(0.0), a);
}

// pconj

template <>
EIGEN_STRONG_INLINE Packet32h pconj<Packet32h>(const Packet32h& a) {
  return a;
}

// psqrt

template <>
EIGEN_STRONG_INLINE Packet32h psqrt<Packet32h>(const Packet32h& a) {
  return _mm512_sqrt_ph(a);
}

// prsqrt

template <>
EIGEN_STRONG_INLINE Packet32h prsqrt<Packet32h>(const Packet32h& a) {
  return _mm512_rsqrt_ph(a);
}

// preciprocal

template <>
EIGEN_STRONG_INLINE Packet32h preciprocal<Packet32h>(const Packet32h& a) {
  return _mm512_rcp_ph(a);
}

// ptranspose

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet32h, 32>& a) {
  __m512i t[32];

  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 16; i++) {
    t[2 * i] = _mm512_unpacklo_epi16(_mm512_castph_si512(a.packet[2 * i]), _mm512_castph_si512(a.packet[2 * i + 1]));
    t[2 * i + 1] =
        _mm512_unpackhi_epi16(_mm512_castph_si512(a.packet[2 * i]), _mm512_castph_si512(a.packet[2 * i + 1]));
  }

  __m512i p[32];

  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 8; i++) {
    p[4 * i] = _mm512_unpacklo_epi32(t[4 * i], t[4 * i + 2]);
    p[4 * i + 1] = _mm512_unpackhi_epi32(t[4 * i], t[4 * i + 2]);
    p[4 * i + 2] = _mm512_unpacklo_epi32(t[4 * i + 1], t[4 * i + 3]);
    p[4 * i + 3] = _mm512_unpackhi_epi32(t[4 * i + 1], t[4 * i + 3]);
  }

  __m512i q[32];

  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 4; i++) {
    q[8 * i] = _mm512_unpacklo_epi64(p[8 * i], p[8 * i + 4]);
    q[8 * i + 1] = _mm512_unpackhi_epi64(p[8 * i], p[8 * i + 4]);
    q[8 * i + 2] = _mm512_unpacklo_epi64(p[8 * i + 1], p[8 * i + 5]);
    q[8 * i + 3] = _mm512_unpackhi_epi64(p[8 * i + 1], p[8 * i + 5]);
    q[8 * i + 4] = _mm512_unpacklo_epi64(p[8 * i + 2], p[8 * i + 6]);
    q[8 * i + 5] = _mm512_unpackhi_epi64(p[8 * i + 2], p[8 * i + 6]);
    q[8 * i + 6] = _mm512_unpacklo_epi64(p[8 * i + 3], p[8 * i + 7]);
    q[8 * i + 7] = _mm512_unpackhi_epi64(p[8 * i + 3], p[8 * i + 7]);
  }

  __m512i f[32];

#define PACKET32H_TRANSPOSE_HELPER(X, Y)                                                            \
  do {                                                                                              \
    f[Y * 8] = _mm512_inserti32x4(f[Y * 8], _mm512_extracti32x4_epi32(q[X * 8], Y), X);             \
    f[Y * 8 + 1] = _mm512_inserti32x4(f[Y * 8 + 1], _mm512_extracti32x4_epi32(q[X * 8 + 1], Y), X); \
    f[Y * 8 + 2] = _mm512_inserti32x4(f[Y * 8 + 2], _mm512_extracti32x4_epi32(q[X * 8 + 2], Y), X); \
    f[Y * 8 + 3] = _mm512_inserti32x4(f[Y * 8 + 3], _mm512_extracti32x4_epi32(q[X * 8 + 3], Y), X); \
    f[Y * 8 + 4] = _mm512_inserti32x4(f[Y * 8 + 4], _mm512_extracti32x4_epi32(q[X * 8 + 4], Y), X); \
    f[Y * 8 + 5] = _mm512_inserti32x4(f[Y * 8 + 5], _mm512_extracti32x4_epi32(q[X * 8 + 5], Y), X); \
    f[Y * 8 + 6] = _mm512_inserti32x4(f[Y * 8 + 6], _mm512_extracti32x4_epi32(q[X * 8 + 6], Y), X); \
    f[Y * 8 + 7] = _mm512_inserti32x4(f[Y * 8 + 7], _mm512_extracti32x4_epi32(q[X * 8 + 7], Y), X); \
  } while (false);

  PACKET32H_TRANSPOSE_HELPER(0, 0);
  PACKET32H_TRANSPOSE_HELPER(1, 1);
  PACKET32H_TRANSPOSE_HELPER(2, 2);
  PACKET32H_TRANSPOSE_HELPER(3, 3);

  PACKET32H_TRANSPOSE_HELPER(1, 0);
  PACKET32H_TRANSPOSE_HELPER(2, 0);
  PACKET32H_TRANSPOSE_HELPER(3, 0);
  PACKET32H_TRANSPOSE_HELPER(2, 1);
  PACKET32H_TRANSPOSE_HELPER(3, 1);
  PACKET32H_TRANSPOSE_HELPER(3, 2);

  PACKET32H_TRANSPOSE_HELPER(0, 1);
  PACKET32H_TRANSPOSE_HELPER(0, 2);
  PACKET32H_TRANSPOSE_HELPER(0, 3);
  PACKET32H_TRANSPOSE_HELPER(1, 2);
  PACKET32H_TRANSPOSE_HELPER(1, 3);
  PACKET32H_TRANSPOSE_HELPER(2, 3);

#undef PACKET32H_TRANSPOSE_HELPER

  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 32; i++) {
    a.packet[i] = _mm512_castsi512_ph(f[i]);
  }
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet32h, 4>& a) {
  __m512i p0, p1, p2, p3, t0, t1, t2, t3, a0, a1, a2, a3;
  t0 = _mm512_unpacklo_epi16(_mm512_castph_si512(a.packet[0]), _mm512_castph_si512(a.packet[1]));
  t1 = _mm512_unpackhi_epi16(_mm512_castph_si512(a.packet[0]), _mm512_castph_si512(a.packet[1]));
  t2 = _mm512_unpacklo_epi16(_mm512_castph_si512(a.packet[2]), _mm512_castph_si512(a.packet[3]));
  t3 = _mm512_unpackhi_epi16(_mm512_castph_si512(a.packet[2]), _mm512_castph_si512(a.packet[3]));

  p0 = _mm512_unpacklo_epi32(t0, t2);
  p1 = _mm512_unpackhi_epi32(t0, t2);
  p2 = _mm512_unpacklo_epi32(t1, t3);
  p3 = _mm512_unpackhi_epi32(t1, t3);

  a0 = p0;
  a1 = p1;
  a2 = p2;
  a3 = p3;

  a0 = _mm512_inserti32x4(a0, _mm512_extracti32x4_epi32(p1, 0), 1);
  a1 = _mm512_inserti32x4(a1, _mm512_extracti32x4_epi32(p0, 1), 0);

  a0 = _mm512_inserti32x4(a0, _mm512_extracti32x4_epi32(p2, 0), 2);
  a2 = _mm512_inserti32x4(a2, _mm512_extracti32x4_epi32(p0, 2), 0);

  a0 = _mm512_inserti32x4(a0, _mm512_extracti32x4_epi32(p3, 0), 3);
  a3 = _mm512_inserti32x4(a3, _mm512_extracti32x4_epi32(p0, 3), 0);

  a1 = _mm512_inserti32x4(a1, _mm512_extracti32x4_epi32(p2, 1), 2);
  a2 = _mm512_inserti32x4(a2, _mm512_extracti32x4_epi32(p1, 2), 1);

  a2 = _mm512_inserti32x4(a2, _mm512_extracti32x4_epi32(p3, 2), 3);
  a3 = _mm512_inserti32x4(a3, _mm512_extracti32x4_epi32(p2, 3), 2);

  a1 = _mm512_inserti32x4(a1, _mm512_extracti32x4_epi32(p3, 1), 3);
  a3 = _mm512_inserti32x4(a3, _mm512_extracti32x4_epi32(p1, 3), 1);

  a.packet[0] = _mm512_castsi512_ph(a0);
  a.packet[1] = _mm512_castsi512_ph(a1);
  a.packet[2] = _mm512_castsi512_ph(a2);
  a.packet[3] = _mm512_castsi512_ph(a3);
}

// preverse

template <>
EIGEN_STRONG_INLINE Packet32h preverse(const Packet32h& a) {
  return _mm512_permutexvar_ph(_mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
                               a);
}

// pscatter

template <>
EIGEN_STRONG_INLINE void pscatter<half, Packet32h>(half* to, const Packet32h& from, Index stride) {
  EIGEN_ALIGN64 half aux[32];
  pstore(aux, from);

  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 32; i++) {
    to[stride * i] = aux[i];
  }
}

// pgather

template <>
EIGEN_STRONG_INLINE Packet32h pgather<Eigen::half, Packet32h>(const Eigen::half* from, Index stride) {
  return _mm512_castsi512_ph(_mm512_set_epi16(
      from[31 * stride].x, from[30 * stride].x, from[29 * stride].x, from[28 * stride].x, from[27 * stride].x,
      from[26 * stride].x, from[25 * stride].x, from[24 * stride].x, from[23 * stride].x, from[22 * stride].x,
      from[21 * stride].x, from[20 * stride].x, from[19 * stride].x, from[18 * stride].x, from[17 * stride].x,
      from[16 * stride].x, from[15 * stride].x, from[14 * stride].x, from[13 * stride].x, from[12 * stride].x,
      from[11 * stride].x, from[10 * stride].x, from[9 * stride].x, from[8 * stride].x, from[7 * stride].x,
      from[6 * stride].x, from[5 * stride].x, from[4 * stride].x, from[3 * stride].x, from[2 * stride].x,
      from[1 * stride].x, from[0 * stride].x));
}

template <>
EIGEN_STRONG_INLINE Packet16h pcos<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h psin<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h plog<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h plog2<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h plog1p<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h pexp<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h pexpm1<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h ptanh<Packet16h>(const Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h pfrexp<Packet16h>(const Packet16h&, Packet16h&);
template <>
EIGEN_STRONG_INLINE Packet16h pldexp<Packet16h>(const Packet16h&, const Packet16h&);

EIGEN_STRONG_INLINE Packet32h combine2Packet16h(const Packet16h& a, const Packet16h& b) {
  __m512d result = _mm512_undefined_pd();
  result = _mm512_insertf64x4(result, _mm256_castsi256_pd(a), 0);
  result = _mm512_insertf64x4(result, _mm256_castsi256_pd(b), 1);
  return _mm512_castpd_ph(result);
}

EIGEN_STRONG_INLINE void extract2Packet16h(const Packet32h& x, Packet16h& a, Packet16h& b) {
  a = _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castph_pd(x), 0));
  b = _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castph_pd(x), 1));
}

// psin
template <>
EIGEN_STRONG_INLINE Packet32h psin<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = psin(low);
  Packet16h highOut = psin(high);

  return combine2Packet16h(lowOut, highOut);
}

// pcos
template <>
EIGEN_STRONG_INLINE Packet32h pcos<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = pcos(low);
  Packet16h highOut = pcos(high);

  return combine2Packet16h(lowOut, highOut);
}

// plog
template <>
EIGEN_STRONG_INLINE Packet32h plog<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = plog(low);
  Packet16h highOut = plog(high);

  return combine2Packet16h(lowOut, highOut);
}

// plog2
template <>
EIGEN_STRONG_INLINE Packet32h plog2<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = plog2(low);
  Packet16h highOut = plog2(high);

  return combine2Packet16h(lowOut, highOut);
}

// plog1p
template <>
EIGEN_STRONG_INLINE Packet32h plog1p<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = plog1p(low);
  Packet16h highOut = plog1p(high);

  return combine2Packet16h(lowOut, highOut);
}

// pexp
template <>
EIGEN_STRONG_INLINE Packet32h pexp<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = pexp(low);
  Packet16h highOut = pexp(high);

  return combine2Packet16h(lowOut, highOut);
}

// pexpm1
template <>
EIGEN_STRONG_INLINE Packet32h pexpm1<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = pexpm1(low);
  Packet16h highOut = pexpm1(high);

  return combine2Packet16h(lowOut, highOut);
}

// ptanh
template <>
EIGEN_STRONG_INLINE Packet32h ptanh<Packet32h>(const Packet32h& a) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h lowOut = ptanh(low);
  Packet16h highOut = ptanh(high);

  return combine2Packet16h(lowOut, highOut);
}

// pfrexp
template <>
EIGEN_STRONG_INLINE Packet32h pfrexp<Packet32h>(const Packet32h& a, Packet32h& exponent) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h exp1 = _mm256_undefined_si256();
  Packet16h exp2 = _mm256_undefined_si256();

  Packet16h lowOut = pfrexp(low, exp1);
  Packet16h highOut = pfrexp(high, exp2);

  exponent = combine2Packet16h(exp1, exp2);

  return combine2Packet16h(lowOut, highOut);
}

// pldexp
template <>
EIGEN_STRONG_INLINE Packet32h pldexp<Packet32h>(const Packet32h& a, const Packet32h& exponent) {
  Packet16h low;
  Packet16h high;
  extract2Packet16h(a, low, high);

  Packet16h exp1;
  Packet16h exp2;
  extract2Packet16h(exponent, exp1, exp2);

  Packet16h lowOut = pldexp(low, exp1);
  Packet16h highOut = pldexp(high, exp2);

  return combine2Packet16h(lowOut, highOut);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_FP16_AVX512_H
