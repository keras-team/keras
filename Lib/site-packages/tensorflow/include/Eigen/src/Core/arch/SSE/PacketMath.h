// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_SSE_H
#define EIGEN_PACKET_MATH_SSE_H

#include <cstdint>
// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#if !defined(EIGEN_VECTORIZE_AVX) && !defined(EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS)
// 32 bits =>  8 registers
// 64 bits => 16 registers
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS (2 * sizeof(void*))
#endif

#ifdef EIGEN_VECTORIZE_FMA
#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif
#endif

#if ((defined EIGEN_VECTORIZE_AVX) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_MINGW || EIGEN_COMP_LCC) && \
     (__GXX_ABI_VERSION < 1004)) ||                                                                     \
    EIGEN_OS_QNX
// With GCC's default ABI version, a __m128 or __m256 are the same types and therefore we cannot
// have overloads for both types without linking error.
// One solution is to increase ABI version using -fabi-version=4 (or greater).
// Otherwise, we workaround this inconvenience by wrapping 128bit types into the following helper
// structure:
typedef eigen_packet_wrapper<__m128> Packet4f;
typedef eigen_packet_wrapper<__m128d> Packet2d;
#else
typedef __m128 Packet4f;
typedef __m128d Packet2d;
#endif

typedef eigen_packet_wrapper<__m128i, 0> Packet4i;
typedef eigen_packet_wrapper<__m128i, 1> Packet16b;
typedef eigen_packet_wrapper<__m128i, 4> Packet4ui;
typedef eigen_packet_wrapper<__m128i, 5> Packet2l;

template <>
struct is_arithmetic<__m128> {
  enum { value = true };
};
template <>
struct is_arithmetic<__m128i> {
  enum { value = true };
};
template <>
struct is_arithmetic<__m128d> {
  enum { value = true };
};
template <>
struct is_arithmetic<Packet4i> {
  enum { value = true };
};
template <>
struct is_arithmetic<Packet2l> {
  enum { value = true };
};
// Note that `Packet4ui` uses the underlying type `__m128i`, which is
// interpreted as a vector of _signed_ `int32`s, which breaks some arithmetic
// operations used in `GenericPacketMath.h`.
template <>
struct is_arithmetic<Packet4ui> {
  enum { value = false };
};
template <>
struct is_arithmetic<Packet16b> {
  enum { value = true };
};

template <int p, int q, int r, int s>
struct shuffle_mask {
  enum { mask = (s) << 6 | (r) << 4 | (q) << 2 | (p) };
};

// TODO: change the implementation of all swizzle* ops from macro to template,
#define vec4f_swizzle1(v, p, q, r, s) \
  Packet4f(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), (shuffle_mask<p, q, r, s>::mask))))

#define vec4i_swizzle1(v, p, q, r, s) Packet4i(_mm_shuffle_epi32(v, (shuffle_mask<p, q, r, s>::mask)))

#define vec4ui_swizzle1(v, p, q, r, s) Packet4ui(vec4i_swizzle1(v, p, q, r, s))

#define vec2d_swizzle1(v, p, q) \
  Packet2d(_mm_castsi128_pd(    \
      _mm_shuffle_epi32(_mm_castpd_si128(v), (shuffle_mask<2 * p, 2 * p + 1, 2 * q, 2 * q + 1>::mask))))

#define vec4f_swizzle2(a, b, p, q, r, s) Packet4f(_mm_shuffle_ps((a), (b), (shuffle_mask<p, q, r, s>::mask)))

#define vec4i_swizzle2(a, b, p, q, r, s) \
  Packet4i(                              \
      _mm_castps_si128((_mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), (shuffle_mask<p, q, r, s>::mask)))))

#define vec4ui_swizzle2(a, b, p, q, r, s) Packet4i(vec4i_swizzle2(a, b, p, q, r, s))

EIGEN_STRONG_INLINE Packet4f vec4f_movelh(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_movelh_ps(a, b));
}
EIGEN_STRONG_INLINE Packet4f vec4f_movehl(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_movehl_ps(a, b));
}
EIGEN_STRONG_INLINE Packet4f vec4f_unpacklo(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_unpacklo_ps(a, b));
}
EIGEN_STRONG_INLINE Packet4f vec4f_unpackhi(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_unpackhi_ps(a, b));
}
#define vec4f_duplane(a, p) vec4f_swizzle2(a, a, p, p, p, p)

#define vec2d_swizzle2(a, b, mask) Packet2d(_mm_shuffle_pd(a, b, mask))

EIGEN_STRONG_INLINE Packet2d vec2d_unpacklo(const Packet2d& a, const Packet2d& b) {
  return Packet2d(_mm_unpacklo_pd(a, b));
}
EIGEN_STRONG_INLINE Packet2d vec2d_unpackhi(const Packet2d& a, const Packet2d& b) {
  return Packet2d(_mm_unpackhi_pd(a, b));
}
#define vec2d_duplane(a, p) vec2d_swizzle2(a, a, (p << 1) | p)

#define EIGEN_DECLARE_CONST_Packet4f(NAME, X) const Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define EIGEN_DECLARE_CONST_Packet2d(NAME, X) const Packet2d p2d_##NAME = pset1<Packet2d>(X)

#define EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME, X) const Packet4f p4f_##NAME = pset1frombits<Packet4f>(X)

#define EIGEN_DECLARE_CONST_Packet4i(NAME, X) const Packet4i p4i_##NAME = pset1<Packet4i>(X)

#define EIGEN_DECLARE_CONST_Packet4ui(NAME, X) const Packet4ui p4ui_##NAME = pset1<Packet4ui>(X)

// Work around lack of extract/cvt for epi64 when compiling for 32-bit.
#if EIGEN_ARCH_x86_64
EIGEN_ALWAYS_INLINE int64_t _mm_extract_epi64_0(const __m128i& a) { return _mm_cvtsi128_si64(a); }
#ifdef EIGEN_VECTORIZE_SSE4_1
EIGEN_ALWAYS_INLINE int64_t _mm_extract_epi64_1(const __m128i& a) { return _mm_extract_epi64(a, 1); }
#else
EIGEN_ALWAYS_INLINE int64_t _mm_extract_epi64_1(const __m128i& a) {
  return _mm_cvtsi128_si64(_mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(a), 0x1)));
}
#endif
#else
// epi64 instructions are not available.  The following seems to generate the same instructions
// with -O2 in GCC/Clang.
EIGEN_ALWAYS_INLINE int64_t _mm_extract_epi64_0(const __m128i& a) {
  return numext::bit_cast<int64_t>(_mm_cvtsd_f64(_mm_castsi128_pd(a)));
}
EIGEN_ALWAYS_INLINE int64_t _mm_extract_epi64_1(const __m128i& a) {
  return numext::bit_cast<int64_t>(_mm_cvtsd_f64(_mm_shuffle_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(a), 0x1)));
}
#endif

// Use the packet_traits defined in AVX/PacketMath.h instead if we're going
// to leverage AVX instructions.
#ifndef EIGEN_VECTORIZE_AVX
template <>
struct packet_traits<float> : default_packet_traits {
  typedef Packet4f type;
  typedef Packet4f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,

    HasCmp = 1,
    HasDiv = 1,
    HasReciprocal = EIGEN_FAST_MATH,
    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasACos = 1,
    HasASin = 1,
    HasATan = 1,
    HasATanh = 1,
    HasLog = 1,
    HasLog1p = 1,
    HasExpm1 = 1,
    HasNdtri = 1,
    HasExp = 1,
    HasBessel = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH,
    HasBlend = 1,
    HasSign = 0  // The manually vectorized version is slightly slower for SSE.
  };
};
template <>
struct packet_traits<double> : default_packet_traits {
  typedef Packet2d type;
  typedef Packet2d half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,

    HasCmp = 1,
    HasDiv = 1,
    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasATan = 1,
    HasBlend = 1
  };
};
template <>
struct packet_traits<int> : default_packet_traits {
  typedef Packet4i type;
  typedef Packet4i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,

    HasCmp = 1,
    HasDiv = 1,
    HasShift = 1,
    HasBlend = 1
  };
};
template <>
struct packet_traits<uint32_t> : default_packet_traits {
  typedef Packet4ui type;
  typedef Packet4ui half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,

    HasDiv = 0,
    HasNegate = 0,
    HasCmp = 1,
    HasShift = 1,
    HasBlend = 1
  };
};
template <>
struct packet_traits<int64_t> : default_packet_traits {
  typedef Packet2l type;
  typedef Packet2l half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,

    HasDiv = 0,
    HasCmp = 1,
    HasShift = 1,
    HasBlend = 1
  };
};
#endif
template <>
struct packet_traits<bool> : default_packet_traits {
  typedef Packet16b type;
  typedef Packet16b half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,

    HasCmp = 1,  // note -- only pcmp_eq is defined
    HasShift = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasConj = 0,
    HasSqrt = 1,
    HasNegate = 0,
    HasSign = 0  // Don't try to vectorize psign<bool> = identity.
  };
};

template <>
struct unpacket_traits<Packet4f> {
  typedef float type;
  typedef Packet4f half;
  typedef Packet4i integer_packet;
  enum {
    size = 4,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet2d> {
  typedef double type;
  typedef Packet2d half;
  typedef Packet2l integer_packet;
  enum {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet2l> {
  typedef int64_t type;
  typedef Packet2l half;
  enum {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet4i> {
  typedef int type;
  typedef Packet4i half;
  enum {
    size = 4,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet4ui> {
  typedef uint32_t type;
  typedef Packet4ui half;
  enum {
    size = 4,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet16b> {
  typedef bool type;
  typedef Packet16b half;
  enum {
    size = 16,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

#ifndef EIGEN_VECTORIZE_AVX
template <>
struct scalar_div_cost<float, true> {
  enum { value = 7 };
};
template <>
struct scalar_div_cost<double, true> {
  enum { value = 8 };
};
#endif

template <>
EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float& from) {
  return _mm_set_ps1(from);
}
template <>
EIGEN_STRONG_INLINE Packet2d pset1<Packet2d>(const double& from) {
  return _mm_set1_pd(from);
}
template <>
EIGEN_STRONG_INLINE Packet2l pset1<Packet2l>(const int64_t& from) {
  return _mm_set1_epi64x(from);
}
template <>
EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int& from) {
  return _mm_set1_epi32(from);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pset1<Packet4ui>(const uint32_t& from) {
  return _mm_set1_epi32(numext::bit_cast<int32_t>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16b pset1<Packet16b>(const bool& from) {
  return _mm_set1_epi8(static_cast<char>(from));
}

template <>
EIGEN_STRONG_INLINE Packet4f pset1frombits<Packet4f>(unsigned int from) {
  return _mm_castsi128_ps(pset1<Packet4i>(from));
}
template <>
EIGEN_STRONG_INLINE Packet2d pset1frombits<Packet2d>(uint64_t from) {
  return _mm_castsi128_pd(_mm_set1_epi64x(from));
}

template <>
EIGEN_STRONG_INLINE Packet4f peven_mask(const Packet4f& /*a*/) {
  return _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1));
}
template <>
EIGEN_STRONG_INLINE Packet2l peven_mask(const Packet2l& /*a*/) {
  return _mm_set_epi32(0, 0, -1, -1);
}
template <>
EIGEN_STRONG_INLINE Packet4i peven_mask(const Packet4i& /*a*/) {
  return _mm_set_epi32(0, -1, 0, -1);
}
template <>
EIGEN_STRONG_INLINE Packet4ui peven_mask(const Packet4ui& /*a*/) {
  return _mm_set_epi32(0, -1, 0, -1);
}
template <>
EIGEN_STRONG_INLINE Packet2d peven_mask(const Packet2d& /*a*/) {
  return _mm_castsi128_pd(_mm_set_epi32(0, 0, -1, -1));
}

template <>
EIGEN_STRONG_INLINE Packet4f pzero(const Packet4f& /*a*/) {
  return _mm_setzero_ps();
}
template <>
EIGEN_STRONG_INLINE Packet2d pzero(const Packet2d& /*a*/) {
  return _mm_setzero_pd();
}
template <>
EIGEN_STRONG_INLINE Packet2l pzero(const Packet2l& /*a*/) {
  return _mm_setzero_si128();
}
template <>
EIGEN_STRONG_INLINE Packet4i pzero(const Packet4i& /*a*/) {
  return _mm_setzero_si128();
}
template <>
EIGEN_STRONG_INLINE Packet4ui pzero(const Packet4ui& /*a*/) {
  return _mm_setzero_si128();
}

// GCC generates a shufps instruction for _mm_set1_ps/_mm_load1_ps instead of the more efficient pshufd instruction.
// However, using inrinsics for pset1 makes gcc to generate crappy code in some cases (see bug 203)
// Using inline assembly is also not an option because then gcc fails to reorder properly the instructions.
// Therefore, we introduced the pload1 functions to be used in product kernels for which bug 203 does not apply.
// Also note that with AVX, we want it to generate a vbroadcastss.
#if EIGEN_COMP_GNUC_STRICT && (!defined __AVX__)
template <>
EIGEN_STRONG_INLINE Packet4f pload1<Packet4f>(const float* from) {
  return vec4f_swizzle1(_mm_load_ss(from), 0, 0, 0, 0);
}
#endif

template <>
EIGEN_STRONG_INLINE Packet4f plset<Packet4f>(const float& a) {
  return _mm_add_ps(pset1<Packet4f>(a), _mm_set_ps(3, 2, 1, 0));
}
template <>
EIGEN_STRONG_INLINE Packet2d plset<Packet2d>(const double& a) {
  return _mm_add_pd(pset1<Packet2d>(a), _mm_set_pd(1, 0));
}
template <>
EIGEN_STRONG_INLINE Packet2l plset<Packet2l>(const int64_t& a) {
  return _mm_add_epi32(pset1<Packet2l>(a), _mm_set_epi64x(1, 0));
}
template <>
EIGEN_STRONG_INLINE Packet4i plset<Packet4i>(const int& a) {
  return _mm_add_epi32(pset1<Packet4i>(a), _mm_set_epi32(3, 2, 1, 0));
}
template <>
EIGEN_STRONG_INLINE Packet4ui plset<Packet4ui>(const uint32_t& a) {
  return _mm_add_epi32(pset1<Packet4ui>(a), _mm_set_epi32(3, 2, 1, 0));
}

template <>
EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_add_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d padd<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_add_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2l padd<Packet2l>(const Packet2l& a, const Packet2l& b) {
  return _mm_add_epi64(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i& a, const Packet4i& b) {
  return _mm_add_epi32(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4ui padd<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
  return _mm_add_epi32(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet16b padd<Packet16b>(const Packet16b& a, const Packet16b& b) {
  return _mm_or_si128(a, b);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet padds(const Packet& a, const Packet& b);
template <>
EIGEN_STRONG_INLINE Packet4f padds<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_add_ss(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d padds<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_add_sd(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_sub_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d psub<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_sub_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2l psub<Packet2l>(const Packet2l& a, const Packet2l& b) {
  return _mm_sub_epi64(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i& a, const Packet4i& b) {
  return _mm_sub_epi32(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4ui psub<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
  return _mm_sub_epi32(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16b psub<Packet16b>(const Packet16b& a, const Packet16b& b) {
  return _mm_xor_si128(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b);
template <>
EIGEN_STRONG_INLINE Packet4f paddsub<Packet4f>(const Packet4f& a, const Packet4f& b) {
#ifdef EIGEN_VECTORIZE_SSE3
  return _mm_addsub_ps(a, b);
#else
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0x0, 0x80000000, 0x0));
  return padd(a, pxor(mask, b));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet2d pxor<Packet2d>(const Packet2d&, const Packet2d&);
template <>
EIGEN_STRONG_INLINE Packet2d paddsub<Packet2d>(const Packet2d& a, const Packet2d& b) {
#ifdef EIGEN_VECTORIZE_SSE3
  return _mm_addsub_pd(a, b);
#else
  const Packet2d mask = _mm_castsi128_pd(_mm_setr_epi32(0x0, 0x80000000, 0x0, 0x0));
  return padd(a, pxor(mask, b));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f& a) {
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
  return _mm_xor_ps(a, mask);
}
template <>
EIGEN_STRONG_INLINE Packet2d pnegate(const Packet2d& a) {
  const Packet2d mask = _mm_castsi128_pd(_mm_setr_epi32(0x0, 0x80000000, 0x0, 0x80000000));
  return _mm_xor_pd(a, mask);
}
template <>
EIGEN_STRONG_INLINE Packet2l pnegate(const Packet2l& a) {
  return psub(pzero(a), a);
}

template <>
EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i& a) {
  return psub(pzero(a), a);
}

template <>
EIGEN_STRONG_INLINE Packet4f pconj(const Packet4f& a) {
  return a;
}
template <>
EIGEN_STRONG_INLINE Packet2d pconj(const Packet2d& a) {
  return a;
}
template <>
EIGEN_STRONG_INLINE Packet2l pconj(const Packet2l& a) {
  return a;
}
template <>
EIGEN_STRONG_INLINE Packet4i pconj(const Packet4i& a) {
  return a;
}

template <>
EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_mul_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmul<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_mul_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2l pmul<Packet2l>(const Packet2l& a, const Packet2l& b) {
  // 64-bit mul requires avx512, so do this with 32-bit multiplication
  __m128i upper32_a = _mm_srli_epi64(a, 32);
  __m128i upper32_b = _mm_srli_epi64(b, 32);

  // upper * lower
  __m128i mul1 = _mm_mul_epu32(upper32_a, b);
  __m128i mul2 = _mm_mul_epu32(upper32_b, a);
  // Gives us both upper*upper and lower*lower
  __m128i mul3 = _mm_mul_epu32(a, b);

  __m128i high = _mm_slli_epi64(_mm_add_epi64(mul1, mul2), 32);
  return _mm_add_epi64(high, mul3);
}
template <>
EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i& a, const Packet4i& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_mullo_epi32(a, b);
#else
  // this version is slightly faster than 4 scalar products
  return vec4i_swizzle1(
      vec4i_swizzle2(_mm_mul_epu32(a, b), _mm_mul_epu32(vec4i_swizzle1(a, 1, 0, 3, 2), vec4i_swizzle1(b, 1, 0, 3, 2)),
                     0, 2, 0, 2),
      0, 2, 1, 3);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet4ui pmul<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_mullo_epi32(a, b);
#else
  // this version is slightly faster than 4 scalar products
  return vec4ui_swizzle1(
      vec4ui_swizzle2(_mm_mul_epu32(a, b),
                      _mm_mul_epu32(vec4ui_swizzle1(a, 1, 0, 3, 2), vec4ui_swizzle1(b, 1, 0, 3, 2)), 0, 2, 0, 2),
      0, 2, 1, 3);
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16b pmul<Packet16b>(const Packet16b& a, const Packet16b& b) {
  return _mm_and_si128(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_div_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pdiv<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_div_pd(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4i pdiv<Packet4i>(const Packet4i& a, const Packet4i& b) {
#ifdef EIGEN_VECTORIZE_AVX
  return _mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(a), _mm256_cvtepi32_pd(b)));
#else
  __m128i q_lo = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(a), _mm_cvtepi32_pd(b)));
  __m128i q_hi = _mm_cvttpd_epi32(
      _mm_div_pd(_mm_cvtepi32_pd(vec4i_swizzle1(a, 2, 3, 0, 1)), _mm_cvtepi32_pd(vec4i_swizzle1(b, 2, 3, 0, 1))));
  return vec4i_swizzle1(_mm_unpacklo_epi32(q_lo, q_hi), 0, 2, 1, 3);
#endif
}

#ifdef EIGEN_VECTORIZE_FMA
template <>
EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c) {
  return _mm_fmadd_ps(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmadd(const Packet2d& a, const Packet2d& b, const Packet2d& c) {
  return _mm_fmadd_pd(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet4f pmsub(const Packet4f& a, const Packet4f& b, const Packet4f& c) {
  return _mm_fmsub_ps(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmsub(const Packet2d& a, const Packet2d& b, const Packet2d& c) {
  return _mm_fmsub_pd(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet4f pnmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c) {
  return _mm_fnmadd_ps(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet2d pnmadd(const Packet2d& a, const Packet2d& b, const Packet2d& c) {
  return _mm_fnmadd_pd(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet4f pnmsub(const Packet4f& a, const Packet4f& b, const Packet4f& c) {
  return _mm_fnmsub_ps(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet2d pnmsub(const Packet2d& a, const Packet2d& b, const Packet2d& c) {
  return _mm_fnmsub_pd(a, b, c);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet pmadds(const Packet& a, const Packet& b, const Packet& c);
template <>
EIGEN_STRONG_INLINE Packet4f pmadds<Packet4f>(const Packet4f& a, const Packet4f& b, const Packet4f& c) {
  return _mm_fmadd_ss(a, b, c);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmadds<Packet2d>(const Packet2d& a, const Packet2d& b, const Packet2d& c) {
  return _mm_fmadd_sd(a, b, c);
}
#endif

#ifdef EIGEN_VECTORIZE_SSE4_1
template <>
EIGEN_STRONG_INLINE Packet4f pselect(const Packet4f& mask, const Packet4f& a, const Packet4f& b) {
  return _mm_blendv_ps(b, a, mask);
}

template <>
EIGEN_STRONG_INLINE Packet2l pselect(const Packet2l& mask, const Packet2l& a, const Packet2l& b) {
  return _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(b), _mm_castsi128_pd(a), _mm_castsi128_pd(mask)));
}

template <>
EIGEN_STRONG_INLINE Packet4i pselect(const Packet4i& mask, const Packet4i& a, const Packet4i& b) {
  return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(a), _mm_castsi128_ps(mask)));
}

template <>
EIGEN_STRONG_INLINE Packet4ui pselect(const Packet4ui& mask, const Packet4ui& a, const Packet4ui& b) {
  return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(a), _mm_castsi128_ps(mask)));
}

template <>
EIGEN_STRONG_INLINE Packet2d pselect(const Packet2d& mask, const Packet2d& a, const Packet2d& b) {
  return _mm_blendv_pd(b, a, mask);
}
#endif

template <>
EIGEN_STRONG_INLINE Packet2l ptrue<Packet2l>(const Packet2l& a) {
  return _mm_cmpeq_epi32(a, a);
}
template <>
EIGEN_STRONG_INLINE Packet4i ptrue<Packet4i>(const Packet4i& a) {
  return _mm_cmpeq_epi32(a, a);
}
template <>
EIGEN_STRONG_INLINE Packet16b ptrue<Packet16b>(const Packet16b& /*a*/) {
  return pset1<Packet16b>(true);
}
template <>
EIGEN_STRONG_INLINE Packet4f ptrue<Packet4f>(const Packet4f& a) {
  Packet4i b = _mm_castps_si128(a);
  return _mm_castsi128_ps(_mm_cmpeq_epi32(b, b));
}
template <>
EIGEN_STRONG_INLINE Packet2d ptrue<Packet2d>(const Packet2d& a) {
  Packet4i b = _mm_castpd_si128(a);
  return _mm_castsi128_pd(_mm_cmpeq_epi32(b, b));
}

template <>
EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_and_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pand<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_and_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2l pand<Packet2l>(const Packet2l& a, const Packet2l& b) {
  return _mm_and_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i& a, const Packet4i& b) {
  return _mm_and_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pand<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
  return _mm_and_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16b pand<Packet16b>(const Packet16b& a, const Packet16b& b) {
  return _mm_and_si128(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_or_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d por<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_or_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2l por<Packet2l>(const Packet2l& a, const Packet2l& b) {
  return _mm_or_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i& a, const Packet4i& b) {
  return _mm_or_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4ui por<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
  return _mm_or_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16b por<Packet16b>(const Packet16b& a, const Packet16b& b) {
  return _mm_or_si128(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_xor_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pxor<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_xor_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2l pxor<Packet2l>(const Packet2l& a, const Packet2l& b) {
  return _mm_xor_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i& a, const Packet4i& b) {
  return _mm_xor_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pxor<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
  return _mm_xor_si128(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16b pxor<Packet16b>(const Packet16b& a, const Packet16b& b) {
  return _mm_xor_si128(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f& a, const Packet4f& b) {
  return _mm_andnot_ps(b, a);
}
template <>
EIGEN_STRONG_INLINE Packet2d pandnot<Packet2d>(const Packet2d& a, const Packet2d& b) {
  return _mm_andnot_pd(b, a);
}
template <>
EIGEN_STRONG_INLINE Packet2l pandnot<Packet2l>(const Packet2l& a, const Packet2l& b) {
  return _mm_andnot_si128(b, a);
}
template <>
EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i& a, const Packet4i& b) {
  return _mm_andnot_si128(b, a);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pandnot<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
  return _mm_andnot_si128(b, a);
}

template <>
EIGEN_STRONG_INLINE Packet4f pcmp_le(const Packet4f& a, const Packet4f& b) {
  return _mm_cmple_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4f pcmp_lt(const Packet4f& a, const Packet4f& b) {
  return _mm_cmplt_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4f pcmp_lt_or_nan(const Packet4f& a, const Packet4f& b) {
  return _mm_cmpnge_ps(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4f pcmp_eq(const Packet4f& a, const Packet4f& b) {
  return _mm_cmpeq_ps(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet2d pcmp_le(const Packet2d& a, const Packet2d& b) {
  return _mm_cmple_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pcmp_lt(const Packet2d& a, const Packet2d& b) {
  return _mm_cmplt_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pcmp_lt_or_nan(const Packet2d& a, const Packet2d& b) {
  return _mm_cmpnge_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet2d pcmp_eq(const Packet2d& a, const Packet2d& b) {
  return _mm_cmpeq_pd(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i pcmp_lt(const Packet4i& a, const Packet4i& b) {
  return _mm_cmplt_epi32(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i pcmp_eq(const Packet4i& a, const Packet4i& b) {
  return _mm_cmpeq_epi32(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet4i pcmp_le(const Packet4i& a, const Packet4i& b) {
  return por(pcmp_lt(a, b), pcmp_eq(a, b));
}
template <>
EIGEN_STRONG_INLINE Packet2l pcmp_lt(const Packet2l& a, const Packet2l& b) {
#ifdef EIGEN_VECTORIZE_SSE4_2
  return _mm_cmpgt_epi64(b, a);
#else
  Packet4i eq = pcmp_eq<Packet4i>(Packet4i(a), Packet4i(b));
  Packet2l hi_eq = Packet2l(_mm_shuffle_epi32(eq, (shuffle_mask<1, 1, 3, 3>::mask)));
  Packet4i lt = pcmp_lt<Packet4i>(Packet4i(a), Packet4i(b));
  Packet2l hi_lt = Packet2l(_mm_shuffle_epi32(lt, (shuffle_mask<1, 1, 3, 3>::mask)));
  Packet2l lo_lt = Packet2l(_mm_shuffle_epi32(lt, (shuffle_mask<0, 0, 2, 2>::mask)));
  // return hi(a) < hi(b) || (hi(a) == hi(b) && lo(a) < lo(b))
  return por(hi_lt, pand(hi_eq, lo_lt));
#endif
}
template <>
EIGEN_STRONG_INLINE Packet2l pcmp_eq(const Packet2l& a, const Packet2l& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_cmpeq_epi64(a, b);
#else
  Packet4i tmp = pcmp_eq<Packet4i>(Packet4i(a), Packet4i(b));
  return Packet2l(pand<Packet4i>(tmp, _mm_shuffle_epi32(tmp, (shuffle_mask<1, 0, 3, 2>::mask))));
#endif
}
template <>
EIGEN_STRONG_INLINE Packet2l pcmp_le(const Packet2l& a, const Packet2l& b) {
  return por(pcmp_lt(a, b), pcmp_eq(a, b));
}
template <>
EIGEN_STRONG_INLINE Packet16b pcmp_eq(const Packet16b& a, const Packet16b& b) {
  // Mask out invalid bool bits to avoid UB.
  const Packet16b kBoolMask = pset1<Packet16b>(true);
  return _mm_and_si128(_mm_cmpeq_epi8(a, b), kBoolMask);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pcmp_eq(const Packet4ui& a, const Packet4ui& b) {
  return _mm_cmpeq_epi32(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f& a, const Packet4f& b) {
#if EIGEN_GNUC_STRICT_LESS_THAN(6, 3, 0)
// There appears to be a bug in GCC, by which the optimizer may
// flip the argument order in calls to _mm_min_ps, so we have to
// resort to inline ASM here. This is supposed to be fixed in gcc6.3,
// see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
#ifdef EIGEN_VECTORIZE_AVX
  Packet4f res;
  asm("vminps %[a], %[b], %[res]" : [res] "=x"(res) : [a] "x"(a), [b] "x"(b));
#else
  Packet4f res = b;
  asm("minps %[a], %[res]" : [res] "+x"(res) : [a] "x"(a));
#endif
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::min.
  return _mm_min_ps(b, a);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet2d pmin<Packet2d>(const Packet2d& a, const Packet2d& b) {
#if EIGEN_GNUC_STRICT_LESS_THAN(6, 3, 0)
// There appears to be a bug in GCC, by which the optimizer may
// flip the argument order in calls to _mm_min_pd, so we have to
// resort to inline ASM here. This is supposed to be fixed in gcc6.3,
// see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
#ifdef EIGEN_VECTORIZE_AVX
  Packet2d res;
  asm("vminpd %[a], %[b], %[res]" : [res] "=x"(res) : [a] "x"(a), [b] "x"(b));
#else
  Packet2d res = b;
  asm("minpd %[a], %[res]" : [res] "+x"(res) : [a] "x"(a));
#endif
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::min.
  return _mm_min_pd(b, a);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet2l pmin<Packet2l>(const Packet2l& a, const Packet2l& b) {
  Packet2l a_lt_mask = pcmp_lt(a, b);
  return por(pandnot(b, a_lt_mask), pand(a, a_lt_mask));
}
template <>
EIGEN_STRONG_INLINE Packet4i pmin<Packet4i>(const Packet4i& a, const Packet4i& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_min_epi32(a, b);
#else
  // after some bench, this version *is* faster than a scalar implementation
  Packet4i mask = _mm_cmplt_epi32(a, b);
  return _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b));
#endif
}
template <>
EIGEN_STRONG_INLINE Packet4ui pmin<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_min_epu32(a, b);
#else
  return padd((Packet4ui)pmin((Packet4i)psub(a, pset1<Packet4ui>(0x80000000UL)),
                              (Packet4i)psub(b, pset1<Packet4ui>(0x80000000UL))),
              pset1<Packet4ui>(0x80000000UL));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f& a, const Packet4f& b) {
#if EIGEN_GNUC_STRICT_LESS_THAN(6, 3, 0)
// There appears to be a bug in GCC, by which the optimizer may
// flip the argument order in calls to _mm_max_ps, so we have to
// resort to inline ASM here. This is supposed to be fixed in gcc6.3,
// see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
#ifdef EIGEN_VECTORIZE_AVX
  Packet4f res;
  asm("vmaxps %[a], %[b], %[res]" : [res] "=x"(res) : [a] "x"(a), [b] "x"(b));
#else
  Packet4f res = b;
  asm("maxps %[a], %[res]" : [res] "+x"(res) : [a] "x"(a));
#endif
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::max.
  return _mm_max_ps(b, a);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet2d pmax<Packet2d>(const Packet2d& a, const Packet2d& b) {
#if EIGEN_GNUC_STRICT_LESS_THAN(6, 3, 0)
// There appears to be a bug in GCC, by which the optimizer may
// flip the argument order in calls to _mm_max_pd, so we have to
// resort to inline ASM here. This is supposed to be fixed in gcc6.3,
// see also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72867
#ifdef EIGEN_VECTORIZE_AVX
  Packet2d res;
  asm("vmaxpd %[a], %[b], %[res]" : [res] "=x"(res) : [a] "x"(a), [b] "x"(b));
#else
  Packet2d res = b;
  asm("maxpd %[a], %[res]" : [res] "+x"(res) : [a] "x"(a));
#endif
  return res;
#else
  // Arguments are reversed to match NaN propagation behavior of std::max.
  return _mm_max_pd(b, a);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet2l pmax<Packet2l>(const Packet2l& a, const Packet2l& b) {
  Packet2l a_lt_mask = pcmp_lt(a, b);
  return por(pandnot(a, a_lt_mask), pand(b, a_lt_mask));
}
template <>
EIGEN_STRONG_INLINE Packet4i pmax<Packet4i>(const Packet4i& a, const Packet4i& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_max_epi32(a, b);
#else
  // after some bench, this version *is* faster than a scalar implementation
  Packet4i mask = _mm_cmpgt_epi32(a, b);
  return _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b));
#endif
}
template <>
EIGEN_STRONG_INLINE Packet4ui pmax<Packet4ui>(const Packet4ui& a, const Packet4ui& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return _mm_max_epu32(a, b);
#else
  return padd((Packet4ui)pmax((Packet4i)psub(a, pset1<Packet4ui>(0x80000000UL)),
                              (Packet4i)psub(b, pset1<Packet4ui>(0x80000000UL))),
              pset1<Packet4ui>(0x80000000UL));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet4ui pcmp_lt(const Packet4ui& a, const Packet4ui& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return pxor(pcmp_eq(a, pmax(a, b)), ptrue(a));
#else
  return (Packet4ui)pcmp_lt((Packet4i)psub(a, pset1<Packet4ui>(0x80000000UL)),
                            (Packet4i)psub(b, pset1<Packet4ui>(0x80000000UL)));
#endif
}
template <>
EIGEN_STRONG_INLINE Packet4ui pcmp_le(const Packet4ui& a, const Packet4ui& b) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  return pcmp_eq(a, pmin(a, b));
#else
  return (Packet4ui)pcmp_le((Packet4i)psub(a, pset1<Packet4ui>(0x80000000UL)),
                            (Packet4i)psub(b, pset1<Packet4ui>(0x80000000UL)));
#endif
}

template <typename Packet, typename Op>
EIGEN_STRONG_INLINE Packet pminmax_propagate_numbers(const Packet& a, const Packet& b, Op op) {
  // In this implementation, we take advantage of the fact that pmin/pmax for SSE
  // always return a if either a or b is NaN.
  Packet not_nan_mask_a = pcmp_eq(a, a);
  Packet m = op(a, b);
  return pselect<Packet>(not_nan_mask_a, m, b);
}

template <typename Packet, typename Op>
EIGEN_STRONG_INLINE Packet pminmax_propagate_nan(const Packet& a, const Packet& b, Op op) {
  // In this implementation, we take advantage of the fact that pmin/pmax for SSE
  // always return a if either a or b is NaN.
  Packet not_nan_mask_a = pcmp_eq(a, a);
  Packet m = op(b, a);
  return pselect<Packet>(not_nan_mask_a, m, a);
}

// Add specializations for min/max with prescribed NaN progation.
template <>
EIGEN_STRONG_INLINE Packet4f pmin<PropagateNumbers, Packet4f>(const Packet4f& a, const Packet4f& b) {
  return pminmax_propagate_numbers(a, b, pmin<Packet4f>);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmin<PropagateNumbers, Packet2d>(const Packet2d& a, const Packet2d& b) {
  return pminmax_propagate_numbers(a, b, pmin<Packet2d>);
}
template <>
EIGEN_STRONG_INLINE Packet4f pmax<PropagateNumbers, Packet4f>(const Packet4f& a, const Packet4f& b) {
  return pminmax_propagate_numbers(a, b, pmax<Packet4f>);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmax<PropagateNumbers, Packet2d>(const Packet2d& a, const Packet2d& b) {
  return pminmax_propagate_numbers(a, b, pmax<Packet2d>);
}
template <>
EIGEN_STRONG_INLINE Packet4f pmin<PropagateNaN, Packet4f>(const Packet4f& a, const Packet4f& b) {
  return pminmax_propagate_nan(a, b, pmin<Packet4f>);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmin<PropagateNaN, Packet2d>(const Packet2d& a, const Packet2d& b) {
  return pminmax_propagate_nan(a, b, pmin<Packet2d>);
}
template <>
EIGEN_STRONG_INLINE Packet4f pmax<PropagateNaN, Packet4f>(const Packet4f& a, const Packet4f& b) {
  return pminmax_propagate_nan(a, b, pmax<Packet4f>);
}
template <>
EIGEN_STRONG_INLINE Packet2d pmax<PropagateNaN, Packet2d>(const Packet2d& a, const Packet2d& b) {
  return pminmax_propagate_nan(a, b, pmax<Packet2d>);
}

template <>
EIGEN_STRONG_INLINE Packet4f psignbit(const Packet4f& a) {
  return _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), 31));
}
template <>
EIGEN_STRONG_INLINE Packet2d psignbit(const Packet2d& a) {
  Packet4f tmp = psignbit<Packet4f>(_mm_castpd_ps(a));
#ifdef EIGEN_VECTORIZE_AVX
  return _mm_castps_pd(_mm_permute_ps(tmp, (shuffle_mask<1, 1, 3, 3>::mask)));
#else
  return _mm_castps_pd(_mm_shuffle_ps(tmp, tmp, (shuffle_mask<1, 1, 3, 3>::mask)));
#endif  // EIGEN_VECTORIZE_AVX
}
template <>
EIGEN_STRONG_INLINE Packet4i psignbit(const Packet4i& a) {
  return _mm_srai_epi32(a, 31);
}
template <>
EIGEN_STRONG_INLINE Packet4ui psignbit(const Packet4ui& a) {
  return pzero(a);
}
template <>
EIGEN_STRONG_INLINE Packet2l psignbit(const Packet2l& a) {
  Packet4i tmp = psignbit<Packet4i>(Packet4i(a));
  return Packet2l(_mm_shuffle_epi32(tmp, (shuffle_mask<1, 1, 3, 3>::mask)));
}

template <int N>
EIGEN_STRONG_INLINE Packet2l parithmetic_shift_right(const Packet2l& a) {
  Packet2l signbit = psignbit(a);
  return por(_mm_slli_epi64(signbit, 64 - N), _mm_srli_epi64(a, N));
}
template <int N>
EIGEN_STRONG_INLINE Packet2l plogical_shift_right(const Packet2l& a) {
  return _mm_srli_epi64(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet2l plogical_shift_left(const Packet2l& a) {
  return _mm_slli_epi64(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet4i parithmetic_shift_right(const Packet4i& a) {
  return _mm_srai_epi32(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet4i plogical_shift_right(const Packet4i& a) {
  return _mm_srli_epi32(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet4i plogical_shift_left(const Packet4i& a) {
  return _mm_slli_epi32(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet4ui parithmetic_shift_right(const Packet4ui& a) {
  return _mm_srli_epi32(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet4ui plogical_shift_right(const Packet4ui& a) {
  return _mm_srli_epi32(a, N);
}
template <int N>
EIGEN_STRONG_INLINE Packet4ui plogical_shift_left(const Packet4ui& a) {
  return _mm_slli_epi32(a, N);
}

template <>
EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f& a) {
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF));
  return _mm_and_ps(a, mask);
}
template <>
EIGEN_STRONG_INLINE Packet2d pabs(const Packet2d& a) {
  const Packet2d mask = _mm_castsi128_pd(_mm_setr_epi32(0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF));
  return _mm_and_pd(a, mask);
}
template <>
EIGEN_STRONG_INLINE Packet2l pabs(const Packet2l& a) {
  Packet2l signbit = psignbit(a);
  return _mm_sub_epi64(_mm_xor_si128(a, signbit), signbit);
}
template <>
EIGEN_STRONG_INLINE Packet4i pabs(const Packet4i& a) {
#ifdef EIGEN_VECTORIZE_SSSE3
  return _mm_abs_epi32(a);
#else
  Packet4i signbit = psignbit(a);
  return _mm_sub_epi32(_mm_xor_si128(a, signbit), signbit);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet4ui pabs(const Packet4ui& a) {
  return a;
}

#ifdef EIGEN_VECTORIZE_SSE4_1
template <>
EIGEN_STRONG_INLINE Packet4f pround<Packet4f>(const Packet4f& a) {
  // Unfortunately _mm_round_ps doesn't have a rounding mode to implement numext::round.
  const Packet4f mask = pset1frombits<Packet4f>(0x80000000u);
  const Packet4f prev0dot5 = pset1frombits<Packet4f>(0x3EFFFFFFu);
  return _mm_round_ps(padd(por(pand(a, mask), prev0dot5), a), _MM_FROUND_TO_ZERO);
}

template <>
EIGEN_STRONG_INLINE Packet2d pround<Packet2d>(const Packet2d& a) {
  const Packet2d mask = _mm_castsi128_pd(_mm_set_epi64x(0x8000000000000000ull, 0x8000000000000000ull));
  const Packet2d prev0dot5 = _mm_castsi128_pd(_mm_set_epi64x(0x3FDFFFFFFFFFFFFFull, 0x3FDFFFFFFFFFFFFFull));
  return _mm_round_pd(padd(por(pand(a, mask), prev0dot5), a), _MM_FROUND_TO_ZERO);
}

template <>
EIGEN_STRONG_INLINE Packet4f print<Packet4f>(const Packet4f& a) {
  return _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
}
template <>
EIGEN_STRONG_INLINE Packet2d print<Packet2d>(const Packet2d& a) {
  return _mm_round_pd(a, _MM_FROUND_CUR_DIRECTION);
}

template <>
EIGEN_STRONG_INLINE Packet4f pceil<Packet4f>(const Packet4f& a) {
  return _mm_ceil_ps(a);
}
template <>
EIGEN_STRONG_INLINE Packet2d pceil<Packet2d>(const Packet2d& a) {
  return _mm_ceil_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f pfloor<Packet4f>(const Packet4f& a) {
  return _mm_floor_ps(a);
}
template <>
EIGEN_STRONG_INLINE Packet2d pfloor<Packet2d>(const Packet2d& a) {
  return _mm_floor_pd(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f ptrunc<Packet4f>(const Packet4f& a) {
  return _mm_round_ps(a, _MM_FROUND_TRUNC);
}
template <>
EIGEN_STRONG_INLINE Packet2d ptrunc<Packet2d>(const Packet2d& a) {
  return _mm_round_pd(a, _MM_FROUND_TRUNC);
}
#endif

template <>
EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_ps(from);
}
template <>
EIGEN_STRONG_INLINE Packet2d pload<Packet2d>(const double* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_pd(from);
}
template <>
EIGEN_STRONG_INLINE Packet2l pload<Packet2l>(const int64_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet4ui pload<Packet4ui>(const uint32_t* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16b pload<Packet16b>(const bool* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(reinterpret_cast<const __m128i*>(from));
}

#if EIGEN_COMP_MSVC
template <>
EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_ps(from);
}
#else
// NOTE: with the code below, MSVC's compiler crashes!

template <>
EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_ps(from);
}
#endif

template <>
EIGEN_STRONG_INLINE Packet2d ploadu<Packet2d>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_pd(from);
}
template <>
EIGEN_STRONG_INLINE Packet2l ploadu<Packet2l>(const int64_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet4ui ploadu<Packet4ui>(const uint32_t* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16b ploadu<Packet16b>(const bool* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
}

// Load lower part of packet zero extending.
template <typename Packet>
EIGEN_STRONG_INLINE Packet ploadl(const typename unpacket_traits<Packet>::type* from);
template <>
EIGEN_STRONG_INLINE Packet4f ploadl<Packet4f>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(from)));
}
template <>
EIGEN_STRONG_INLINE Packet2d ploadl<Packet2d>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_load_sd(from);
}

// Load scalar
template <typename Packet>
EIGEN_STRONG_INLINE Packet ploads(const typename unpacket_traits<Packet>::type* from);
template <>
EIGEN_STRONG_INLINE Packet4f ploads<Packet4f>(const float* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_load_ss(from);
}
template <>
EIGEN_STRONG_INLINE Packet2d ploads<Packet2d>(const double* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_load_sd(from);
}

template <>
EIGEN_STRONG_INLINE Packet4f ploaddup<Packet4f>(const float* from) {
  return vec4f_swizzle1(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(from))), 0, 0, 1, 1);
}
template <>
EIGEN_STRONG_INLINE Packet2d ploaddup<Packet2d>(const double* from) {
  return pset1<Packet2d>(from[0]);
}
template <>
EIGEN_STRONG_INLINE Packet2l ploaddup<Packet2l>(const int64_t* from) {
  return pset1<Packet2l>(from[0]);
}
template <>
EIGEN_STRONG_INLINE Packet4i ploaddup<Packet4i>(const int* from) {
  Packet4i tmp;
  tmp = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from));
  return vec4i_swizzle1(tmp, 0, 0, 1, 1);
}
template <>
EIGEN_STRONG_INLINE Packet4ui ploaddup<Packet4ui>(const uint32_t* from) {
  Packet4ui tmp;
  tmp = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from));
  return vec4ui_swizzle1(tmp, 0, 0, 1, 1);
}

// Loads 8 bools from memory and returns the packet
// {b0, b0, b1, b1, b2, b2, b3, b3, b4, b4, b5, b5, b6, b6, b7, b7}
template <>
EIGEN_STRONG_INLINE Packet16b ploaddup<Packet16b>(const bool* from) {
  __m128i tmp = _mm_castpd_si128(pload1<Packet2d>(reinterpret_cast<const double*>(from)));
  return _mm_unpacklo_epi8(tmp, tmp);
}

// Loads 4 bools from memory and returns the packet
// {b0, b0  b0, b0, b1, b1, b1, b1, b2, b2, b2, b2, b3, b3, b3, b3}
template <>
EIGEN_STRONG_INLINE Packet16b ploadquad<Packet16b>(const bool* from) {
  __m128i tmp = _mm_castps_si128(pload1<Packet4f>(reinterpret_cast<const float*>(from)));
  tmp = _mm_unpacklo_epi8(tmp, tmp);
  return _mm_unpacklo_epi16(tmp, tmp);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet4f& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_ps(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstore<double>(double* to, const Packet2d& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_pd(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstore<int64_t>(int64_t* to, const Packet2l& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstore<int>(int* to, const Packet4i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstore<uint32_t>(uint32_t* to, const Packet4ui& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstore<bool>(bool* to, const Packet16b& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to), from);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packet2d& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_pd(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet4f& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_ps(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<int64_t>(int64_t* to, const Packet2l& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<int>(int* to, const Packet4i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<uint32_t>(uint32_t* to, const Packet4ui& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<bool>(bool* to, const Packet16b& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from);
}

template <typename Scalar, typename Packet>
EIGEN_STRONG_INLINE void pstorel(Scalar* to, const Packet& from);
template <>
EIGEN_STRONG_INLINE void pstorel(float* to, const Packet4f& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storel_pi(reinterpret_cast<__m64*>(to), from);
}
template <>
EIGEN_STRONG_INLINE void pstorel(double* to, const Packet2d& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storel_pd(to, from);
}

template <typename Scalar, typename Packet>
EIGEN_STRONG_INLINE void pstores(Scalar* to, const Packet& from);
template <>
EIGEN_STRONG_INLINE void pstores(float* to, const Packet4f& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_store_ss(to, from);
}
template <>
EIGEN_STRONG_INLINE void pstores(double* to, const Packet2d& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_store_sd(to, from);
}

template <>
EIGEN_STRONG_INLINE Packet4f preverse(const Packet4f& a) {
  return _mm_shuffle_ps(a, a, 0x1B);
}
template <>
EIGEN_STRONG_INLINE Packet2d preverse(const Packet2d& a) {
  return _mm_shuffle_pd(a, a, 0x1);
}
template <>
EIGEN_STRONG_INLINE Packet2l preverse(const Packet2l& a) {
  return _mm_castpd_si128(preverse(_mm_castsi128_pd(a)));
}
template <>
EIGEN_STRONG_INLINE Packet4i preverse(const Packet4i& a) {
  return _mm_shuffle_epi32(a, 0x1B);
}
template <>
EIGEN_STRONG_INLINE Packet4ui preverse(const Packet4ui& a) {
  return _mm_shuffle_epi32(a, 0x1B);
}
template <>
EIGEN_STRONG_INLINE Packet16b preverse(const Packet16b& a) {
#ifdef EIGEN_VECTORIZE_SSSE3
  __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  return _mm_shuffle_epi8(a, mask);
#else
  Packet16b tmp = _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 1, 2, 3));
  tmp = _mm_shufflehi_epi16(_mm_shufflelo_epi16(tmp, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_or_si128(_mm_slli_epi16(tmp, 8), _mm_srli_epi16(tmp, 8));
#endif
}

#if EIGEN_COMP_MSVC_STRICT && EIGEN_OS_WIN64
// The temporary variable fixes an internal compilation error in vs <= 2008 and a wrong-result bug in vs 2010
// Direct of the struct members fixed bug #62.
template <>
EIGEN_STRONG_INLINE float pfirst<Packet4f>(const Packet4f& a) {
  return a.m128_f32[0];
}
template <>
EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) {
  return a.m128d_f64[0];
}
template <>
EIGEN_STRONG_INLINE int64_t pfirst<Packet2l>(const Packet2l& a) {
  int64_t x = _mm_extract_epi64_0(a);
  return x;
}
template <>
EIGEN_STRONG_INLINE int pfirst<Packet4i>(const Packet4i& a) {
  int x = _mm_cvtsi128_si32(a);
  return x;
}
template <>
EIGEN_STRONG_INLINE uint32_t pfirst<Packet4ui>(const Packet4ui& a) {
  uint32_t x = numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(a));
  return x;
}
#elif EIGEN_COMP_MSVC_STRICT
// The temporary variable fixes an internal compilation error in vs <= 2008 and a wrong-result bug in vs 2010
template <>
EIGEN_STRONG_INLINE float pfirst<Packet4f>(const Packet4f& a) {
  float x = _mm_cvtss_f32(a);
  return x;
}
template <>
EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) {
  double x = _mm_cvtsd_f64(a);
  return x;
}
template <>
EIGEN_STRONG_INLINE int64_t pfirst<Packet2l>(const Packet2l& a) {
  int64_t x = _mm_extract_epi64_0(a);
  return x;
}
template <>
EIGEN_STRONG_INLINE int pfirst<Packet4i>(const Packet4i& a) {
  int x = _mm_cvtsi128_si32(a);
  return x;
}
template <>
EIGEN_STRONG_INLINE uint32_t pfirst<Packet4ui>(const Packet4ui& a) {
  uint32_t x = numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(a));
  return x;
}
#else
template <>
EIGEN_STRONG_INLINE float pfirst<Packet4f>(const Packet4f& a) {
  return _mm_cvtss_f32(a);
}
template <>
EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) {
  return _mm_cvtsd_f64(a);
}
template <>
EIGEN_STRONG_INLINE int64_t pfirst<Packet2l>(const Packet2l& a) {
  return _mm_extract_epi64_0(a);
}
template <>
EIGEN_STRONG_INLINE int pfirst<Packet4i>(const Packet4i& a) {
  return _mm_cvtsi128_si32(a);
}
template <>
EIGEN_STRONG_INLINE uint32_t pfirst<Packet4ui>(const Packet4ui& a) {
  return numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(a));
}
#endif
template <>
EIGEN_STRONG_INLINE bool pfirst<Packet16b>(const Packet16b& a) {
  int x = _mm_cvtsi128_si32(a);
  return static_cast<bool>(x & 1);
}

template <>
EIGEN_STRONG_INLINE Packet4f pgather<float, Packet4f>(const float* from, Index stride) {
  return _mm_set_ps(from[3 * stride], from[2 * stride], from[1 * stride], from[0 * stride]);
}
template <>
EIGEN_STRONG_INLINE Packet2d pgather<double, Packet2d>(const double* from, Index stride) {
  return _mm_set_pd(from[1 * stride], from[0 * stride]);
}
template <>
EIGEN_STRONG_INLINE Packet2l pgather<int64_t, Packet2l>(const int64_t* from, Index stride) {
  return _mm_set_epi64x(from[1 * stride], from[0 * stride]);
}
template <>
EIGEN_STRONG_INLINE Packet4i pgather<int, Packet4i>(const int* from, Index stride) {
  return _mm_set_epi32(from[3 * stride], from[2 * stride], from[1 * stride], from[0 * stride]);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pgather<uint32_t, Packet4ui>(const uint32_t* from, Index stride) {
  return _mm_set_epi32(numext::bit_cast<int32_t>(from[3 * stride]), numext::bit_cast<int32_t>(from[2 * stride]),
                       numext::bit_cast<int32_t>(from[1 * stride]), numext::bit_cast<int32_t>(from[0 * stride]));
}

template <>
EIGEN_STRONG_INLINE Packet16b pgather<bool, Packet16b>(const bool* from, Index stride) {
  return _mm_set_epi8(from[15 * stride], from[14 * stride], from[13 * stride], from[12 * stride], from[11 * stride],
                      from[10 * stride], from[9 * stride], from[8 * stride], from[7 * stride], from[6 * stride],
                      from[5 * stride], from[4 * stride], from[3 * stride], from[2 * stride], from[1 * stride],
                      from[0 * stride]);
}

template <>
EIGEN_STRONG_INLINE void pscatter<float, Packet4f>(float* to, const Packet4f& from, Index stride) {
  to[stride * 0] = pfirst(from);
  to[stride * 1] = pfirst(_mm_shuffle_ps(from, from, 1));
  to[stride * 2] = pfirst(_mm_shuffle_ps(from, from, 2));
  to[stride * 3] = pfirst(_mm_shuffle_ps(from, from, 3));
}
template <>
EIGEN_STRONG_INLINE void pscatter<double, Packet2d>(double* to, const Packet2d& from, Index stride) {
  to[stride * 0] = pfirst(from);
  to[stride * 1] = pfirst(preverse(from));
}
template <>
EIGEN_STRONG_INLINE void pscatter<int64_t, Packet2l>(int64_t* to, const Packet2l& from, Index stride) {
  to[stride * 0] = pfirst(from);
  to[stride * 1] = pfirst(preverse(from));
}
template <>
EIGEN_STRONG_INLINE void pscatter<int, Packet4i>(int* to, const Packet4i& from, Index stride) {
  to[stride * 0] = _mm_cvtsi128_si32(from);
  to[stride * 1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 1));
  to[stride * 2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 2));
  to[stride * 3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 3));
}
template <>
EIGEN_STRONG_INLINE void pscatter<uint32_t, Packet4ui>(uint32_t* to, const Packet4ui& from, Index stride) {
  to[stride * 0] = numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(from));
  to[stride * 1] = numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi32(from, 1)));
  to[stride * 2] = numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi32(from, 2)));
  to[stride * 3] = numext::bit_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi32(from, 3)));
}
template <>
EIGEN_STRONG_INLINE void pscatter<bool, Packet16b>(bool* to, const Packet16b& from, Index stride) {
  to[4 * stride * 0] = _mm_cvtsi128_si32(from);
  to[4 * stride * 1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 1));
  to[4 * stride * 2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 2));
  to[4 * stride * 3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(from, 3));
}

// some compilers might be tempted to perform multiple moves instead of using a vector path.
template <>
EIGEN_STRONG_INLINE void pstore1<Packet4f>(float* to, const float& a) {
  Packet4f pa = _mm_set_ss(a);
  pstore(to, Packet4f(vec4f_swizzle1(pa, 0, 0, 0, 0)));
}
// some compilers might be tempted to perform multiple moves instead of using a vector path.
template <>
EIGEN_STRONG_INLINE void pstore1<Packet2d>(double* to, const double& a) {
  Packet2d pa = _mm_set_sd(a);
  pstore(to, Packet2d(vec2d_swizzle1(pa, 0, 0)));
}

#if EIGEN_COMP_PGI && EIGEN_COMP_PGI < 1900
typedef const void* SsePrefetchPtrType;
#else
typedef const char* SsePrefetchPtrType;
#endif

#ifndef EIGEN_VECTORIZE_AVX
template <>
EIGEN_STRONG_INLINE void prefetch<float>(const float* addr) {
  _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0);
}
template <>
EIGEN_STRONG_INLINE void prefetch<double>(const double* addr) {
  _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0);
}
template <>
EIGEN_STRONG_INLINE void prefetch<int>(const int* addr) {
  _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0);
}
template <>
EIGEN_STRONG_INLINE void prefetch<int64_t>(const int64_t* addr) {
  _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0);
}
template <>
EIGEN_STRONG_INLINE void prefetch<uint32_t>(const uint32_t* addr) {
  _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0);
}
#endif

template <>
EIGEN_STRONG_INLINE Packet4f pfrexp<Packet4f>(const Packet4f& a, Packet4f& exponent) {
  return pfrexp_generic(a, exponent);
}

// Extract exponent without existence of Packet2l.
template <>
EIGEN_STRONG_INLINE Packet2d pfrexp_generic_get_biased_exponent(const Packet2d& a) {
  const Packet2d cst_exp_mask = pset1frombits<Packet2d>(static_cast<uint64_t>(0x7ff0000000000000ull));
  __m128i a_expo = _mm_srli_epi64(_mm_castpd_si128(pand(a, cst_exp_mask)), 52);
  return _mm_cvtepi32_pd(vec4i_swizzle1(a_expo, 0, 2, 1, 3));
}

template <>
EIGEN_STRONG_INLINE Packet2d pfrexp<Packet2d>(const Packet2d& a, Packet2d& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE Packet4f pldexp<Packet4f>(const Packet4f& a, const Packet4f& exponent) {
  return pldexp_generic(a, exponent);
}

// We specialize pldexp here, since the generic implementation uses Packet2l, which is not well
// supported by SSE, and has more range than is needed for exponents.
// TODO(rmlarsen): Remove this specialization once Packet2l has support or casting.
template <>
EIGEN_STRONG_INLINE Packet2d pldexp<Packet2d>(const Packet2d& a, const Packet2d& exponent) {
  // Clamp exponent to [-2099, 2099]
  const Packet2d max_exponent = pset1<Packet2d>(2099.0);
  const Packet2d e = pmin(pmax(exponent, pnegate(max_exponent)), max_exponent);

  // Convert e to integer and swizzle to low-order bits.
  const Packet4i ei = vec4i_swizzle1(_mm_cvtpd_epi32(e), 0, 3, 1, 3);

  // Split 2^e into four factors and multiply:
  const Packet4i bias = _mm_set_epi32(0, 1023, 0, 1023);
  Packet4i b = parithmetic_shift_right<2>(ei);                       // floor(e/4)
  Packet2d c = _mm_castsi128_pd(_mm_slli_epi64(padd(b, bias), 52));  // 2^b
  Packet2d out = pmul(pmul(pmul(a, c), c), c);                       // a * 2^(3b)
  b = psub(psub(psub(ei, b), b), b);                                 // e - 3b
  c = _mm_castsi128_pd(_mm_slli_epi64(padd(b, bias), 52));           // 2^(e - 3b)
  out = pmul(out, c);                                                // a * 2^e
  return out;
}

// with AVX, the default implementations based on pload1 are faster
#ifndef __AVX__
template <>
EIGEN_STRONG_INLINE void pbroadcast4<Packet4f>(const float* a, Packet4f& a0, Packet4f& a1, Packet4f& a2, Packet4f& a3) {
  a3 = pload<Packet4f>(a);
  a0 = vec4f_swizzle1(a3, 0, 0, 0, 0);
  a1 = vec4f_swizzle1(a3, 1, 1, 1, 1);
  a2 = vec4f_swizzle1(a3, 2, 2, 2, 2);
  a3 = vec4f_swizzle1(a3, 3, 3, 3, 3);
}
template <>
EIGEN_STRONG_INLINE void pbroadcast4<Packet2d>(const double* a, Packet2d& a0, Packet2d& a1, Packet2d& a2,
                                               Packet2d& a3) {
#ifdef EIGEN_VECTORIZE_SSE3
  a0 = _mm_loaddup_pd(a + 0);
  a1 = _mm_loaddup_pd(a + 1);
  a2 = _mm_loaddup_pd(a + 2);
  a3 = _mm_loaddup_pd(a + 3);
#else
  a1 = pload<Packet2d>(a);
  a0 = vec2d_swizzle1(a1, 0, 0);
  a1 = vec2d_swizzle1(a1, 1, 1);
  a3 = pload<Packet2d>(a + 2);
  a2 = vec2d_swizzle1(a3, 0, 0);
  a3 = vec2d_swizzle1(a3, 1, 1);
#endif
}
#endif

EIGEN_STRONG_INLINE void punpackp(Packet4f* vecs) {
  vecs[1] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0x55));
  vecs[2] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0xAA));
  vecs[3] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0xFF));
  vecs[0] = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vecs[0]), 0x00));
}

template <>
EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a) {
  // Disable SSE3 _mm_hadd_pd that is extremely slow on all existing Intel's architectures
  // (from Nehalem to Haswell)
  // #ifdef EIGEN_VECTORIZE_SSE3
  //   Packet4f tmp = _mm_add_ps(a, vec4f_swizzle1(a,2,3,2,3));
  //   return pfirst<Packet4f>(_mm_hadd_ps(tmp, tmp));
  // #else
  Packet4f tmp = _mm_add_ps(a, _mm_movehl_ps(a, a));
  return pfirst<Packet4f>(_mm_add_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1)));
  // #endif
}

template <>
EIGEN_STRONG_INLINE double predux<Packet2d>(const Packet2d& a) {
  // Disable SSE3 _mm_hadd_pd that is extremely slow on all existing Intel's architectures
  // (from Nehalem to Haswell)
  // #ifdef EIGEN_VECTORIZE_SSE3
  //   return pfirst<Packet2d>(_mm_hadd_pd(a, a));
  // #else
  return pfirst<Packet2d>(_mm_add_sd(a, _mm_unpackhi_pd(a, a)));
  // #endif
}

template <>
EIGEN_STRONG_INLINE int64_t predux<Packet2l>(const Packet2l& a) {
  return pfirst<Packet2l>(_mm_add_epi64(a, _mm_unpackhi_epi64(a, a)));
}

#ifdef EIGEN_VECTORIZE_SSSE3
template <>
EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a) {
  Packet4i tmp0 = _mm_hadd_epi32(a, a);
  return pfirst<Packet4i>(_mm_hadd_epi32(tmp0, tmp0));
}
template <>
EIGEN_STRONG_INLINE uint32_t predux<Packet4ui>(const Packet4ui& a) {
  Packet4ui tmp0 = _mm_hadd_epi32(a, a);
  return pfirst<Packet4ui>(_mm_hadd_epi32(tmp0, tmp0));
}
#else
template <>
EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a) {
  Packet4i tmp = _mm_add_epi32(a, _mm_unpackhi_epi64(a, a));
  return pfirst(tmp) + pfirst<Packet4i>(_mm_shuffle_epi32(tmp, 1));
}
template <>
EIGEN_STRONG_INLINE uint32_t predux<Packet4ui>(const Packet4ui& a) {
  Packet4ui tmp = _mm_add_epi32(a, _mm_unpackhi_epi64(a, a));
  return pfirst(tmp) + pfirst<Packet4ui>(_mm_shuffle_epi32(tmp, 1));
}
#endif

template <>
EIGEN_STRONG_INLINE bool predux<Packet16b>(const Packet16b& a) {
  Packet4i tmp = _mm_or_si128(a, _mm_unpackhi_epi64(a, a));
  return (pfirst(tmp) != 0) || (pfirst<Packet4i>(_mm_shuffle_epi32(tmp, 1)) != 0);
}

// Other reduction functions:

// mul
template <>
EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a) {
  Packet4f tmp = _mm_mul_ps(a, _mm_movehl_ps(a, a));
  return pfirst<Packet4f>(_mm_mul_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1)));
}
template <>
EIGEN_STRONG_INLINE double predux_mul<Packet2d>(const Packet2d& a) {
  return pfirst<Packet2d>(_mm_mul_sd(a, _mm_unpackhi_pd(a, a)));
}
template <>
EIGEN_STRONG_INLINE int64_t predux_mul<Packet2l>(const Packet2l& a) {
  EIGEN_ALIGN16 int64_t aux[2];
  pstore(aux, a);
  return aux[0] * aux[1];
}
template <>
EIGEN_STRONG_INLINE int predux_mul<Packet4i>(const Packet4i& a) {
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (e.g., reusing pmul is very slow!)
  // TODO try to call _mm_mul_epu32 directly
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  return (aux[0] * aux[1]) * (aux[2] * aux[3]);
}
template <>
EIGEN_STRONG_INLINE uint32_t predux_mul<Packet4ui>(const Packet4ui& a) {
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., reusing pmul is very slow !)
  // TODO try to call _mm_mul_epu32 directly
  EIGEN_ALIGN16 uint32_t aux[4];
  pstore(aux, a);
  return (aux[0] * aux[1]) * (aux[2] * aux[3]);
}

template <>
EIGEN_STRONG_INLINE bool predux_mul<Packet16b>(const Packet16b& a) {
  Packet4i tmp = _mm_and_si128(a, _mm_unpackhi_epi64(a, a));
  return ((pfirst<Packet4i>(tmp) == 0x01010101) && (pfirst<Packet4i>(_mm_shuffle_epi32(tmp, 1)) == 0x01010101));
}

// min
template <>
EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a) {
  Packet4f tmp = _mm_min_ps(a, _mm_movehl_ps(a, a));
  return pfirst<Packet4f>(_mm_min_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1)));
}
template <>
EIGEN_STRONG_INLINE double predux_min<Packet2d>(const Packet2d& a) {
  return pfirst<Packet2d>(_mm_min_sd(a, _mm_unpackhi_pd(a, a)));
}
template <>
EIGEN_STRONG_INLINE int predux_min<Packet4i>(const Packet4i& a) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  Packet4i tmp = _mm_min_epi32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst<Packet4i>(_mm_min_epi32(tmp, _mm_shuffle_epi32(tmp, 1)));
#else
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., it does not like using std::min after the pstore !!)
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  int aux0 = aux[0] < aux[1] ? aux[0] : aux[1];
  int aux2 = aux[2] < aux[3] ? aux[2] : aux[3];
  return aux0 < aux2 ? aux0 : aux2;
#endif  // EIGEN_VECTORIZE_SSE4_1
}
template <>
EIGEN_STRONG_INLINE uint32_t predux_min<Packet4ui>(const Packet4ui& a) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  Packet4ui tmp = _mm_min_epu32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst<Packet4ui>(_mm_min_epu32(tmp, _mm_shuffle_epi32(tmp, 1)));
#else
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., it does not like using std::min after the pstore !!)
  EIGEN_ALIGN16 uint32_t aux[4];
  pstore(aux, a);
  uint32_t aux0 = aux[0] < aux[1] ? aux[0] : aux[1];
  uint32_t aux2 = aux[2] < aux[3] ? aux[2] : aux[3];
  return aux0 < aux2 ? aux0 : aux2;
#endif  // EIGEN_VECTORIZE_SSE4_1
}

// max
template <>
EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a) {
  Packet4f tmp = _mm_max_ps(a, _mm_movehl_ps(a, a));
  return pfirst<Packet4f>(_mm_max_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1)));
}
template <>
EIGEN_STRONG_INLINE double predux_max<Packet2d>(const Packet2d& a) {
  return pfirst<Packet2d>(_mm_max_sd(a, _mm_unpackhi_pd(a, a)));
}
template <>
EIGEN_STRONG_INLINE int predux_max<Packet4i>(const Packet4i& a) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  Packet4i tmp = _mm_max_epi32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst<Packet4i>(_mm_max_epi32(tmp, _mm_shuffle_epi32(tmp, 1)));
#else
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., it does not like using std::min after the pstore !!)
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  int aux0 = aux[0] > aux[1] ? aux[0] : aux[1];
  int aux2 = aux[2] > aux[3] ? aux[2] : aux[3];
  return aux0 > aux2 ? aux0 : aux2;
#endif  // EIGEN_VECTORIZE_SSE4_1
}
template <>
EIGEN_STRONG_INLINE uint32_t predux_max<Packet4ui>(const Packet4ui& a) {
#ifdef EIGEN_VECTORIZE_SSE4_1
  Packet4ui tmp = _mm_max_epu32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));
  return pfirst<Packet4ui>(_mm_max_epu32(tmp, _mm_shuffle_epi32(tmp, 1)));
#else
  // after some experiments, it is seems this is the fastest way to implement it
  // for GCC (eg., it does not like using std::min after the pstore !!)
  EIGEN_ALIGN16 uint32_t aux[4];
  pstore(aux, a);
  uint32_t aux0 = aux[0] > aux[1] ? aux[0] : aux[1];
  uint32_t aux2 = aux[2] > aux[3] ? aux[2] : aux[3];
  return aux0 > aux2 ? aux0 : aux2;
#endif  // EIGEN_VECTORIZE_SSE4_1
}

// not needed yet
// template<> EIGEN_STRONG_INLINE bool predux_all(const Packet4f& x)
// {
//   return _mm_movemask_ps(x) == 0xF;
// }

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet2d& x) {
  return _mm_movemask_pd(x) != 0x0;
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4f& x) {
  return _mm_movemask_ps(x) != 0x0;
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet2l& x) {
  return _mm_movemask_pd(_mm_castsi128_pd(x)) != 0x0;
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4i& x) {
  return _mm_movemask_ps(_mm_castsi128_ps(x)) != 0x0;
}
template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet4ui& x) {
  return _mm_movemask_ps(_mm_castsi128_ps(x)) != 0x0;
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4f, 4>& kernel) {
  _MM_TRANSPOSE4_PS(kernel.packet[0], kernel.packet[1], kernel.packet[2], kernel.packet[3]);
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet2d, 2>& kernel) {
  __m128d tmp = _mm_unpackhi_pd(kernel.packet[0], kernel.packet[1]);
  kernel.packet[0] = _mm_unpacklo_pd(kernel.packet[0], kernel.packet[1]);
  kernel.packet[1] = tmp;
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet2l, 2>& kernel) {
  __m128i tmp = _mm_unpackhi_epi64(kernel.packet[0], kernel.packet[1]);
  kernel.packet[0] = _mm_unpacklo_epi64(kernel.packet[0], kernel.packet[1]);
  kernel.packet[1] = tmp;
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4i, 4>& kernel) {
  __m128i T0 = _mm_unpacklo_epi32(kernel.packet[0], kernel.packet[1]);
  __m128i T1 = _mm_unpacklo_epi32(kernel.packet[2], kernel.packet[3]);
  __m128i T2 = _mm_unpackhi_epi32(kernel.packet[0], kernel.packet[1]);
  __m128i T3 = _mm_unpackhi_epi32(kernel.packet[2], kernel.packet[3]);

  kernel.packet[0] = _mm_unpacklo_epi64(T0, T1);
  kernel.packet[1] = _mm_unpackhi_epi64(T0, T1);
  kernel.packet[2] = _mm_unpacklo_epi64(T2, T3);
  kernel.packet[3] = _mm_unpackhi_epi64(T2, T3);
}
EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4ui, 4>& kernel) {
  ptranspose((PacketBlock<Packet4i, 4>&)kernel);
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16b, 4>& kernel) {
  __m128i T0 = _mm_unpacklo_epi8(kernel.packet[0], kernel.packet[1]);
  __m128i T1 = _mm_unpackhi_epi8(kernel.packet[0], kernel.packet[1]);
  __m128i T2 = _mm_unpacklo_epi8(kernel.packet[2], kernel.packet[3]);
  __m128i T3 = _mm_unpackhi_epi8(kernel.packet[2], kernel.packet[3]);
  kernel.packet[0] = _mm_unpacklo_epi16(T0, T2);
  kernel.packet[1] = _mm_unpackhi_epi16(T0, T2);
  kernel.packet[2] = _mm_unpacklo_epi16(T1, T3);
  kernel.packet[3] = _mm_unpackhi_epi16(T1, T3);
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16b, 16>& kernel) {
  // If we number the elements in the input thus:
  // kernel.packet[ 0] = {00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c, 0d, 0e, 0f}
  // kernel.packet[ 1] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1a, 1b, 1c, 1d, 1e, 1f}
  // ...
  // kernel.packet[15] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fa, fb, fc, fd, fe, ff},
  //
  // the desired output is:
  // kernel.packet[ 0] = {00, 10, 20, 30, 40, 50, 60, 70, 80, 90, a0, b0, c0, d0, e0, f0}
  // kernel.packet[ 1] = {01, 11, 21, 31, 41, 51, 61, 71, 81, 91, a1, b1, c1, d1, e1, f1}
  // ...
  // kernel.packet[15] = {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, af, bf, cf, df, ef, ff},
  __m128i t0 =
      _mm_unpacklo_epi8(kernel.packet[0], kernel.packet[1]);  // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
  __m128i t1 =
      _mm_unpackhi_epi8(kernel.packet[0], kernel.packet[1]);  // 08 18 09 19 0a 1a 0b 1b 0c 1c 0d 1d 0e 1e 0f 1f
  __m128i t2 =
      _mm_unpacklo_epi8(kernel.packet[2], kernel.packet[3]);  // 20 30 21 31 22 32 ...                     27 37
  __m128i t3 =
      _mm_unpackhi_epi8(kernel.packet[2], kernel.packet[3]);  // 28 38 29 39 2a 3a ...                     2f 3f
  __m128i t4 =
      _mm_unpacklo_epi8(kernel.packet[4], kernel.packet[5]);  // 40 50 41 51 42 52                         47 57
  __m128i t5 = _mm_unpackhi_epi8(kernel.packet[4], kernel.packet[5]);  // 48 58 49 59 4a 5a
  __m128i t6 = _mm_unpacklo_epi8(kernel.packet[6], kernel.packet[7]);
  __m128i t7 = _mm_unpackhi_epi8(kernel.packet[6], kernel.packet[7]);
  __m128i t8 = _mm_unpacklo_epi8(kernel.packet[8], kernel.packet[9]);
  __m128i t9 = _mm_unpackhi_epi8(kernel.packet[8], kernel.packet[9]);
  __m128i ta = _mm_unpacklo_epi8(kernel.packet[10], kernel.packet[11]);
  __m128i tb = _mm_unpackhi_epi8(kernel.packet[10], kernel.packet[11]);
  __m128i tc = _mm_unpacklo_epi8(kernel.packet[12], kernel.packet[13]);
  __m128i td = _mm_unpackhi_epi8(kernel.packet[12], kernel.packet[13]);
  __m128i te = _mm_unpacklo_epi8(kernel.packet[14], kernel.packet[15]);
  __m128i tf = _mm_unpackhi_epi8(kernel.packet[14], kernel.packet[15]);

  __m128i s0 = _mm_unpacklo_epi16(t0, t2);  // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
  __m128i s1 = _mm_unpackhi_epi16(t0, t2);  // 04 14 24 34
  __m128i s2 = _mm_unpacklo_epi16(t1, t3);  // 08 18 28 38 ...
  __m128i s3 = _mm_unpackhi_epi16(t1, t3);  // 0c 1c 2c 3c ...
  __m128i s4 = _mm_unpacklo_epi16(t4, t6);  // 40 50 60 70 41 51 61 71 42 52 62 72 43 53 63 73
  __m128i s5 = _mm_unpackhi_epi16(t4, t6);  // 44 54 64 74 ...
  __m128i s6 = _mm_unpacklo_epi16(t5, t7);
  __m128i s7 = _mm_unpackhi_epi16(t5, t7);
  __m128i s8 = _mm_unpacklo_epi16(t8, ta);
  __m128i s9 = _mm_unpackhi_epi16(t8, ta);
  __m128i sa = _mm_unpacklo_epi16(t9, tb);
  __m128i sb = _mm_unpackhi_epi16(t9, tb);
  __m128i sc = _mm_unpacklo_epi16(tc, te);
  __m128i sd = _mm_unpackhi_epi16(tc, te);
  __m128i se = _mm_unpacklo_epi16(td, tf);
  __m128i sf = _mm_unpackhi_epi16(td, tf);

  __m128i u0 = _mm_unpacklo_epi32(s0, s4);  // 00 10 20 30 40 50 60 70 01 11 21 31 41 51 61 71
  __m128i u1 = _mm_unpackhi_epi32(s0, s4);  // 02 12 22 32 42 52 62 72 03 13 23 33 43 53 63 73
  __m128i u2 = _mm_unpacklo_epi32(s1, s5);
  __m128i u3 = _mm_unpackhi_epi32(s1, s5);
  __m128i u4 = _mm_unpacklo_epi32(s2, s6);
  __m128i u5 = _mm_unpackhi_epi32(s2, s6);
  __m128i u6 = _mm_unpacklo_epi32(s3, s7);
  __m128i u7 = _mm_unpackhi_epi32(s3, s7);
  __m128i u8 = _mm_unpacklo_epi32(s8, sc);
  __m128i u9 = _mm_unpackhi_epi32(s8, sc);
  __m128i ua = _mm_unpacklo_epi32(s9, sd);
  __m128i ub = _mm_unpackhi_epi32(s9, sd);
  __m128i uc = _mm_unpacklo_epi32(sa, se);
  __m128i ud = _mm_unpackhi_epi32(sa, se);
  __m128i ue = _mm_unpacklo_epi32(sb, sf);
  __m128i uf = _mm_unpackhi_epi32(sb, sf);

  kernel.packet[0] = _mm_unpacklo_epi64(u0, u8);
  kernel.packet[1] = _mm_unpackhi_epi64(u0, u8);
  kernel.packet[2] = _mm_unpacklo_epi64(u1, u9);
  kernel.packet[3] = _mm_unpackhi_epi64(u1, u9);
  kernel.packet[4] = _mm_unpacklo_epi64(u2, ua);
  kernel.packet[5] = _mm_unpackhi_epi64(u2, ua);
  kernel.packet[6] = _mm_unpacklo_epi64(u3, ub);
  kernel.packet[7] = _mm_unpackhi_epi64(u3, ub);
  kernel.packet[8] = _mm_unpacklo_epi64(u4, uc);
  kernel.packet[9] = _mm_unpackhi_epi64(u4, uc);
  kernel.packet[10] = _mm_unpacklo_epi64(u5, ud);
  kernel.packet[11] = _mm_unpackhi_epi64(u5, ud);
  kernel.packet[12] = _mm_unpacklo_epi64(u6, ue);
  kernel.packet[13] = _mm_unpackhi_epi64(u6, ue);
  kernel.packet[14] = _mm_unpacklo_epi64(u7, uf);
  kernel.packet[15] = _mm_unpackhi_epi64(u7, uf);
}

EIGEN_STRONG_INLINE __m128i sse_blend_mask(const Selector<2>& ifPacket) {
  return _mm_set_epi64x(0 - ifPacket.select[1], 0 - ifPacket.select[0]);
}

EIGEN_STRONG_INLINE __m128i sse_blend_mask(const Selector<4>& ifPacket) {
  return _mm_set_epi32(0 - ifPacket.select[3], 0 - ifPacket.select[2], 0 - ifPacket.select[1], 0 - ifPacket.select[0]);
}

template <>
EIGEN_STRONG_INLINE Packet2l pblend(const Selector<2>& ifPacket, const Packet2l& thenPacket,
                                    const Packet2l& elsePacket) {
  const __m128i true_mask = sse_blend_mask(ifPacket);
  return pselect<Packet2l>(true_mask, thenPacket, elsePacket);
}
template <>
EIGEN_STRONG_INLINE Packet4i pblend(const Selector<4>& ifPacket, const Packet4i& thenPacket,
                                    const Packet4i& elsePacket) {
  const __m128i true_mask = sse_blend_mask(ifPacket);
  return pselect<Packet4i>(true_mask, thenPacket, elsePacket);
}
template <>
EIGEN_STRONG_INLINE Packet4ui pblend(const Selector<4>& ifPacket, const Packet4ui& thenPacket,
                                     const Packet4ui& elsePacket) {
  return (Packet4ui)pblend(ifPacket, (Packet4i)thenPacket, (Packet4i)elsePacket);
}
template <>
EIGEN_STRONG_INLINE Packet4f pblend(const Selector<4>& ifPacket, const Packet4f& thenPacket,
                                    const Packet4f& elsePacket) {
  const __m128i true_mask = sse_blend_mask(ifPacket);
  return pselect<Packet4f>(_mm_castsi128_ps(true_mask), thenPacket, elsePacket);
}
template <>
EIGEN_STRONG_INLINE Packet2d pblend(const Selector<2>& ifPacket, const Packet2d& thenPacket,
                                    const Packet2d& elsePacket) {
  const __m128i true_mask = sse_blend_mask(ifPacket);
  return pselect<Packet2d>(_mm_castsi128_pd(true_mask), thenPacket, elsePacket);
}

// Scalar path for pmadd with FMA to ensure consistency with vectorized path.
#ifdef EIGEN_VECTORIZE_FMA
template <>
EIGEN_STRONG_INLINE float pmadd(const float& a, const float& b, const float& c) {
  return ::fmaf(a, b, c);
}
template <>
EIGEN_STRONG_INLINE double pmadd(const double& a, const double& b, const double& c) {
  return ::fma(a, b, c);
}
template <>
EIGEN_STRONG_INLINE float pmsub(const float& a, const float& b, const float& c) {
  return ::fmaf(a, b, -c);
}
template <>
EIGEN_STRONG_INLINE double pmsub(const double& a, const double& b, const double& c) {
  return ::fma(a, b, -c);
}
template <>
EIGEN_STRONG_INLINE float pnmadd(const float& a, const float& b, const float& c) {
  return ::fmaf(-a, b, c);
}
template <>
EIGEN_STRONG_INLINE double pnmadd(const double& a, const double& b, const double& c) {
  return ::fma(-a, b, c);
}
template <>
EIGEN_STRONG_INLINE float pnmsub(const float& a, const float& b, const float& c) {
  return ::fmaf(-a, b, -c);
}
template <>
EIGEN_STRONG_INLINE double pnmsub(const double& a, const double& b, const double& c) {
  return ::fma(-a, b, -c);
}
#endif

#ifdef EIGEN_VECTORIZE_SSE4_1
// Helpers for half->float and float->half conversions.
// Currently only used by the AVX code.
EIGEN_STRONG_INLINE __m128i half2floatsse(__m128i h) {
  __m128i input = _mm_cvtepu16_epi32(h);

  // Direct vectorization of half_to_float, C parts in the comments.
  __m128i shifted_exp = _mm_set1_epi32(0x7c00 << 13);
  // o.u = (h.x & 0x7fff) << 13; // exponent/mantissa bits
  __m128i ou = _mm_slli_epi32(_mm_and_si128(input, _mm_set1_epi32(0x7fff)), 13);
  // exp = shifted_exp & o.u;   // just the exponent
  __m128i exp = _mm_and_si128(ou, shifted_exp);
  // o.u += (127 - 15) << 23;
  ou = _mm_add_epi32(ou, _mm_set1_epi32((127 - 15) << 23));

  // Inf/NaN?
  __m128i naninf_mask = _mm_cmpeq_epi32(exp, shifted_exp);
  // Inf/NaN adjust
  __m128i naninf_adj = _mm_and_si128(_mm_set1_epi32((128 - 16) << 23), naninf_mask);
  // extra exp adjust for  Inf/NaN
  ou = _mm_add_epi32(ou, naninf_adj);

  // Zero/Denormal?
  __m128i zeroden_mask = _mm_cmpeq_epi32(exp, _mm_setzero_si128());
  __m128i zeroden_adj = _mm_and_si128(zeroden_mask, _mm_set1_epi32(1 << 23));
  // o.u += 1 << 23;
  ou = _mm_add_epi32(ou, zeroden_adj);
  // magic.u = 113 << 23
  __m128i magic = _mm_and_si128(zeroden_mask, _mm_set1_epi32(113 << 23));
  // o.f -= magic.f
  ou = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(ou), _mm_castsi128_ps(magic)));

  __m128i sign = _mm_slli_epi32(_mm_and_si128(input, _mm_set1_epi32(0x8000)), 16);
  // o.u |= (h.x & 0x8000) << 16;    // sign bit
  ou = _mm_or_si128(ou, sign);
  // return o.f;
  // We are actually returning uint version, to make
  // _mm256_insertf128_si256 work.
  return ou;
}

EIGEN_STRONG_INLINE __m128i float2half(__m128 f) {
  __m128i o = _mm_setzero_si128();

  // unsigned int sign_mask = 0x80000000u;
  __m128i sign = _mm_set1_epi32(0x80000000u);
  // unsigned int sign = f.u & sign_mask;
  sign = _mm_and_si128(sign, _mm_castps_si128(f));
  // f.u ^= sign;
  f = _mm_xor_ps(f, _mm_castsi128_ps(sign));

  __m128i fu = _mm_castps_si128(f);

  __m128i f16max = _mm_set1_epi32((127 + 16) << 23);
  __m128i f32infty = _mm_set1_epi32(255 << 23);
  // if (f.u >= f16max.u) // result is Inf or NaN (all exponent bits set)
  // there is no _mm_cmpge_epi32, so use lt and swap operands
  __m128i infnan_mask = _mm_cmplt_epi32(f16max, _mm_castps_si128(f));
  __m128i inf_mask = _mm_cmpgt_epi32(_mm_castps_si128(f), f32infty);
  __m128i nan_mask = _mm_andnot_si128(inf_mask, infnan_mask);
  __m128i inf_value = _mm_and_si128(inf_mask, _mm_set1_epi32(0x7e00));
  __m128i nan_value = _mm_and_si128(nan_mask, _mm_set1_epi32(0x7c00));
  // o.x = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
  __m128i naninf_value = _mm_or_si128(inf_value, nan_value);

  __m128i denorm_magic = _mm_set1_epi32(((127 - 15) + (23 - 10) + 1) << 23);
  __m128i subnorm_mask = _mm_cmplt_epi32(_mm_castps_si128(f), _mm_set1_epi32(113 << 23));
  //  f.f += denorm_magic.f;
  f = _mm_add_ps(f, _mm_castsi128_ps(denorm_magic));
  // f.u - denorm_magic.u
  o = _mm_sub_epi32(_mm_castps_si128(f), denorm_magic);
  o = _mm_and_si128(o, subnorm_mask);
  // Correct result for inf/nan/zero/subnormal, 0 otherwise
  o = _mm_or_si128(o, naninf_value);

  __m128i mask = _mm_or_si128(infnan_mask, subnorm_mask);
  o = _mm_and_si128(o, mask);

  // mant_odd = (f.u >> 13) & 1;
  __m128i mand_odd = _mm_and_si128(_mm_srli_epi32(fu, 13), _mm_set1_epi32(0x1));
  // f.u += 0xc8000fffU;
  fu = _mm_add_epi32(fu, _mm_set1_epi32(0xc8000fffU));
  // f.u += mant_odd;
  fu = _mm_add_epi32(fu, mand_odd);
  fu = _mm_andnot_si128(mask, fu);
  // f.u >> 13
  fu = _mm_srli_epi32(fu, 13);
  o = _mm_or_si128(fu, o);

  // o.x |= static_cast<numext::uint16_t>(sign >> 16);
  o = _mm_or_si128(o, _mm_srli_epi32(sign, 16));

  // 16 bit values
  return _mm_and_si128(o, _mm_set1_epi32(0xffff));
}
#endif

// Packet math for Eigen::half
// Disable the following code since it's broken on too many platforms / compilers.
// #elif defined(EIGEN_VECTORIZE_SSE) && (!EIGEN_ARCH_x86_64) && (!EIGEN_COMP_MSVC)
#if 0

typedef struct {
  __m64 x;
} Packet4h;


template<> struct is_arithmetic<Packet4h> { enum { value = true }; };

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet4h type;
  // There is no half-size packet for Packet4h.
  typedef Packet4h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 0,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
  };
};


template<> struct unpacket_traits<Packet4h> { typedef Eigen::half type; enum {size=4, alignment=Aligned16, vectorizable=true, masked_load_available=false, masked_store_available=false}; typedef Packet4h half; };

template<> EIGEN_STRONG_INLINE Packet4h pset1<Packet4h>(const Eigen::half& from) {
  Packet4h result;
  result.x = _mm_set1_pi16(from.x);
  return result;
}

template<> EIGEN_STRONG_INLINE Eigen::half pfirst<Packet4h>(const Packet4h& from) {
  return half_impl::raw_uint16_to_half(static_cast<unsigned short>(_mm_cvtsi64_si32(from.x)));
}

template<> EIGEN_STRONG_INLINE Packet4h pconj(const Packet4h& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4h padd<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha + hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h psub<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha - hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha - hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha - hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha - hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pmul<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha * hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pdiv<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha / hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha / hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha / hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha / hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pload<Packet4h>(const Eigen::half* from) {
  Packet4h result;
  result.x = _mm_cvtsi64_m64(*reinterpret_cast<const __int64_t*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h ploadu<Packet4h>(const Eigen::half* from) {
  Packet4h result;
  result.x = _mm_cvtsi64_m64(*reinterpret_cast<const __int64_t*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet4h& from) {
  __int64_t r = _mm_cvtm64_si64(from.x);
  *(reinterpret_cast<__int64_t*>(to)) = r;
}

template<> EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet4h& from) {
  __int64_t r = _mm_cvtm64_si64(from.x);
  *(reinterpret_cast<__int64_t*>(to)) = r;
}

template<> EIGEN_STRONG_INLINE Packet4h
ploadquad<Packet4h>(const Eigen::half* from) {
  return pset1<Packet4h>(*from);
}

template<> EIGEN_STRONG_INLINE Packet4h pgather<Eigen::half, Packet4h>(const Eigen::half* from, Index stride)
{
  Packet4h result;
  result.x = _mm_set_pi16(from[3*stride].x, from[2*stride].x, from[1*stride].x, from[0*stride].x);
  return result;
}

template<> EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet4h>(Eigen::half* to, const Packet4h& from, Index stride)
{
  __int64_t a = _mm_cvtm64_si64(from.x);
  to[stride*0].x = static_cast<unsigned short>(a);
  to[stride*1].x = static_cast<unsigned short>(a >> 16);
  to[stride*2].x = static_cast<unsigned short>(a >> 32);
  to[stride*3].x = static_cast<unsigned short>(a >> 48);
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet4h,4>& kernel) {
  __m64 T0 = _mm_unpacklo_pi16(kernel.packet[0].x, kernel.packet[1].x);
  __m64 T1 = _mm_unpacklo_pi16(kernel.packet[2].x, kernel.packet[3].x);
  __m64 T2 = _mm_unpackhi_pi16(kernel.packet[0].x, kernel.packet[1].x);
  __m64 T3 = _mm_unpackhi_pi16(kernel.packet[2].x, kernel.packet[3].x);

  kernel.packet[0].x = _mm_unpacklo_pi32(T0, T1);
  kernel.packet[1].x = _mm_unpackhi_pi32(T0, T1);
  kernel.packet[2].x = _mm_unpacklo_pi32(T2, T3);
  kernel.packet[3].x = _mm_unpackhi_pi32(T2, T3);
}

#endif

}  // end namespace internal

}  // end namespace Eigen

#if EIGEN_COMP_PGI && EIGEN_COMP_PGI < 1900
// PGI++ does not define the following intrinsics in C++ mode.
static inline __m128 _mm_castpd_ps(__m128d x) { return reinterpret_cast<__m128&>(x); }
static inline __m128i _mm_castpd_si128(__m128d x) { return reinterpret_cast<__m128i&>(x); }
static inline __m128d _mm_castps_pd(__m128 x) { return reinterpret_cast<__m128d&>(x); }
static inline __m128i _mm_castps_si128(__m128 x) { return reinterpret_cast<__m128i&>(x); }
static inline __m128 _mm_castsi128_ps(__m128i x) { return reinterpret_cast<__m128&>(x); }
static inline __m128d _mm_castsi128_pd(__m128i x) { return reinterpret_cast<__m128d&>(x); }
#endif

#endif  // EIGEN_PACKET_MATH_SSE_H
