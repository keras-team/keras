// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// fixedpoint_SSE.h: optimized SSE specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_SSE_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_SSE_H_

#include <smmintrin.h>
#include "fixedpoint.h"

namespace gemmlowp {

// SSE intrinsics are not finely typed: there is a single __m128i vector
// type that does not distinguish between "int32x4" and "int16x8" use
// cases, unlike the NEON equivalents. Because we had initially focused
// on int32x4, we did not pay attention and specialized these fixedpoint
// templates directly for __m128i hardcoding the int32x4 semantics,
// not leaving room for int16x8 semantics. Amending that by adding a separate
// data type, int16x8_m128i, that wraps __m128i while being a separate
// type.
struct int16x8_m128i {
  __m128i v;
};

// Keep int16x8_m128i trivially constructible/destructible and provide
// easily optimized helper function.
inline int16x8_m128i to_int16x8_m128i(__m128i w) {
  int16x8_m128i r;
  r.v = w;
  return r;
}

template <>
struct FixedPointRawTypeTraits<__m128i> {
  typedef std::int32_t ScalarRawType;
  static constexpr int kLanes = 4;
};

template <>
struct FixedPointRawTypeTraits<int16x8_m128i> {
  typedef std::int16_t ScalarRawType;
  static constexpr int kLanes = 8;
};

template <>
inline __m128i BitAnd(__m128i a, __m128i b) {
  return _mm_and_si128(a, b);
}

template <>
inline int16x8_m128i BitAnd(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_and_si128(a.v, b.v));
}

template <>
inline __m128i BitOr(__m128i a, __m128i b) {
  return _mm_or_si128(a, b);
}

template <>
inline int16x8_m128i BitOr(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_or_si128(a.v, b.v));
}

template <>
inline __m128i BitXor(__m128i a, __m128i b) {
  return _mm_xor_si128(a, b);
}

template <>
inline int16x8_m128i BitXor(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_xor_si128(a.v, b.v));
}

template <>
inline __m128i BitNot(__m128i a) {
  return _mm_andnot_si128(a, _mm_set1_epi32(-1));
}

template <>
inline int16x8_m128i BitNot(int16x8_m128i a) {
  return to_int16x8_m128i(_mm_andnot_si128(a.v, _mm_set1_epi16(-1)));
}

template <>
inline __m128i Add(__m128i a, __m128i b) {
  return _mm_add_epi32(a, b);
}

template <>
inline int16x8_m128i Add(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_add_epi16(a.v, b.v));
}

template <>
inline __m128i Mul(__m128i a, __m128i b) {
  return _mm_mullo_epi32(a, b);
}

template <>
inline int16x8_m128i Mul(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_mullo_epi16(a.v, b.v));
}

template <>
inline __m128i Sub(__m128i a, __m128i b) {
  return _mm_sub_epi32(a, b);
}

template <>
inline int16x8_m128i Sub(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_sub_epi16(a.v, b.v));
}

template <>
inline __m128i Neg(__m128i a) {
  return _mm_sign_epi32(a, _mm_set1_epi32(-1));
}

template <>
inline int16x8_m128i Neg(int16x8_m128i a) {
  return to_int16x8_m128i(_mm_sign_epi16(a.v, _mm_set1_epi16(-1)));
}

template <>
inline __m128i ShiftLeft(__m128i a, int offset) {
  return _mm_slli_epi32(a, offset);
}

template <>
inline int16x8_m128i ShiftLeft(int16x8_m128i a, int offset) {
  return to_int16x8_m128i(_mm_slli_epi16(a.v, offset));
}

template <>
inline __m128i ShiftRight(__m128i a, int offset) {
  return _mm_srai_epi32(a, offset);
}

template <>
inline int16x8_m128i ShiftRight(int16x8_m128i a, int offset) {
  return to_int16x8_m128i(_mm_srai_epi16(a.v, offset));
}

template <>
inline __m128i SelectUsingMask(__m128i if_mask, __m128i then_val,
                               __m128i else_val) {
  // borrowed from Intel's arm_neon_sse.h header.
  return _mm_or_si128(_mm_and_si128(if_mask, then_val),
                      _mm_andnot_si128(if_mask, else_val));
}

template <>
inline int16x8_m128i SelectUsingMask(int16x8_m128i if_mask,
                                     int16x8_m128i then_val,
                                     int16x8_m128i else_val) {
  // borrowed from Intel's arm_neon_sse.h header.
  return to_int16x8_m128i(SelectUsingMask(if_mask.v, then_val.v, else_val.v));
}

template <>
inline __m128i MaskIfEqual(__m128i a, __m128i b) {
  return _mm_cmpeq_epi32(a, b);
}

template <>
inline int16x8_m128i MaskIfEqual(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_cmpeq_epi16(a.v, b.v));
}

template <>
inline __m128i MaskIfNotEqual(__m128i a, __m128i b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline int16x8_m128i MaskIfNotEqual(int16x8_m128i a, int16x8_m128i b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline __m128i MaskIfZero(__m128i a) {
  return MaskIfEqual(a, _mm_set1_epi32(0));
}

template <>
inline int16x8_m128i MaskIfZero(int16x8_m128i a) {
  return MaskIfEqual(a, to_int16x8_m128i(_mm_set1_epi16(0)));
}

template <>
inline __m128i MaskIfNonZero(__m128i a) {
  return MaskIfNotEqual(a, _mm_set1_epi32(0));
}

template <>
inline int16x8_m128i MaskIfNonZero(int16x8_m128i a) {
  return MaskIfNotEqual(a, to_int16x8_m128i(_mm_set1_epi16(0)));
}

template <>
inline __m128i MaskIfGreaterThan(__m128i a, __m128i b) {
  return _mm_cmpgt_epi32(a, b);
}

template <>
inline int16x8_m128i MaskIfGreaterThan(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_cmpgt_epi16(a.v, b.v));
}

template <>
inline __m128i MaskIfLessThan(__m128i a, __m128i b) {
  return _mm_cmplt_epi32(a, b);
}

template <>
inline int16x8_m128i MaskIfLessThan(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_cmplt_epi16(a.v, b.v));
}

template <>
inline __m128i MaskIfGreaterThanOrEqual(__m128i a, __m128i b) {
  return BitNot(MaskIfLessThan(a, b));
}

template <>
inline int16x8_m128i MaskIfGreaterThanOrEqual(int16x8_m128i a,
                                              int16x8_m128i b) {
  return BitNot(MaskIfLessThan(a, b));
}

template <>
inline __m128i MaskIfLessThanOrEqual(__m128i a, __m128i b) {
  return BitNot(MaskIfGreaterThan(a, b));
}

template <>
inline int16x8_m128i MaskIfLessThanOrEqual(int16x8_m128i a, int16x8_m128i b) {
  return BitNot(MaskIfGreaterThan(a, b));
}

/* Assumptions:
   - All and Any are used on masks.
   - masks are all_ones for true lanes, all_zeroes otherwise.
Hence, All means all 128bits set, and Any means any bit set.
*/

template <>
inline bool All(__m128i a) {
  return _mm_testc_si128(a, a);
}

template <>
inline bool All(int16x8_m128i a) {
  return _mm_testc_si128(a.v, a.v);
}

template <>
inline bool Any(__m128i a) {
  return !_mm_testz_si128(a, a);
}

template <>
inline bool Any(int16x8_m128i a) {
  return !_mm_testz_si128(a.v, a.v);
}

template <>
inline __m128i RoundingHalfSum(__m128i a, __m128i b) {
  /* __m128i round_bit_mask, a_over_2, b_over_2, round_bit, sum; */
  /* We divide the inputs before the add to avoid the overflow and costly test
   */
  /* of checking if an overflow occured on signed add */
  /* round_bit_mask = _mm_set1_epi32(1); */
  /* a_over_2 = _mm_srai_epi32(a, 1); */
  /* b_over_2 = _mm_srai_epi32(b, 1); */
  /* sum = Add(a_over_2, b_over_2); */
  /* round_bit = _mm_sign_epi32(BitAnd(BitOr(a,b), round_bit_mask), sum); */
  /* return Add(sum, round_bit); */

  /* Other possibility detecting overflow and xor the sign if an overflow
   * happened*/
  __m128i one, sign_bit_mask, sum, rounded_half_sum, overflow, result;
  one = _mm_set1_epi32(1);
  sign_bit_mask = _mm_set1_epi32(0x80000000);
  sum = Add(a, b);
  rounded_half_sum = _mm_srai_epi32(Add(sum, one), 1);
  overflow =
      BitAnd(BitAnd(BitXor(a, rounded_half_sum), BitXor(b, rounded_half_sum)),
             sign_bit_mask);
  result = BitXor(rounded_half_sum, overflow);
  return result;
}

template <>
inline int16x8_m128i RoundingHalfSum(int16x8_m128i a, int16x8_m128i b) {
  // Idea: go to unsigned to use _mm_avg_epu16,
  // borrowed from Intel's arm_neon_sse.h header.
  __m128i constant_neg_32768 = _mm_set1_epi16(-32768);
  __m128i a_unsigned = _mm_sub_epi16(a.v, constant_neg_32768);
  __m128i b_unsigned = _mm_sub_epi16(b.v, constant_neg_32768);
  __m128i avg_unsigned = _mm_avg_epu16(a_unsigned, b_unsigned);
  __m128i avg = _mm_add_epi16(avg_unsigned, constant_neg_32768);
  return to_int16x8_m128i(avg);
}

template <>
inline __m128i SaturatingRoundingDoublingHighMul(__m128i a, __m128i b) {
  __m128i min, saturation_mask, a0_a2, a1_a3, b0_b2, b1_b3;
  __m128i a0b0_a2b2, a1b1_a3b3, a0b0_a2b2_rounded, a1b1_a3b3_rounded;
  __m128i a0b0_a2b2_rounded_2x, a1b1_a3b3_rounded_2x, result;
  __m128i nudge;

  // saturation only happen if a == b == INT_MIN
  min = _mm_set1_epi32(std::numeric_limits<std::int32_t>::min());
  saturation_mask = BitAnd(MaskIfEqual(a, b), MaskIfEqual(a, min));

  // a = a0 | a1 | a2 | a3
  // b = b0 | b1 | b2 | b3
  a0_a2 = a;
  a1_a3 = _mm_srli_si128(a, 4);
  b0_b2 = b;
  b1_b3 = _mm_srli_si128(b, 4);

  a0b0_a2b2 = _mm_mul_epi32(a0_a2, b0_b2);
  a1b1_a3b3 = _mm_mul_epi32(a1_a3, b1_b3);

  // do the rounding and take into account that it will be doubled
  nudge = _mm_set1_epi64x(1 << 30);
  a0b0_a2b2_rounded = _mm_add_epi64(a0b0_a2b2, nudge);
  a1b1_a3b3_rounded = _mm_add_epi64(a1b1_a3b3, nudge);

  // do the doubling
  a0b0_a2b2_rounded_2x = _mm_slli_epi64(a0b0_a2b2_rounded, 1);
  a1b1_a3b3_rounded_2x = _mm_slli_epi64(a1b1_a3b3_rounded, 1);

  // get the high part of the products
  result = _mm_blend_epi16(_mm_srli_si128(a0b0_a2b2_rounded_2x, 4),
                           a1b1_a3b3_rounded_2x, 0xcc);

  // saturate those which overflowed
  return SelectUsingMask(saturation_mask, min, result);
}

template <>
inline int16x8_m128i SaturatingRoundingDoublingHighMul(int16x8_m128i a,
                                                       int16x8_m128i b) {
  // Idea: use _mm_mulhrs_epi16 then saturate with a bit-operation,
  // borrowed from Intel's arm_neon_sse.h header.
  __m128i result_unsaturated = _mm_mulhrs_epi16(a.v, b.v);
  __m128i saturation_mask =
      _mm_cmpeq_epi16(result_unsaturated, _mm_set1_epi16(0x8000));
  __m128i result = _mm_xor_si128(result_unsaturated, saturation_mask);
  return to_int16x8_m128i(result);
}

template <>
inline __m128i Dup<__m128i>(std::int32_t x) {
  return _mm_set1_epi32(x);
}

template <>
inline int16x8_m128i Dup<int16x8_m128i>(std::int16_t x) {
  return to_int16x8_m128i(_mm_set1_epi16(x));
}

// So far this is only needed for int16.
template <>
inline int16x8_m128i SaturatingAdd(int16x8_m128i a, int16x8_m128i b) {
  return to_int16x8_m128i(_mm_adds_epi16(a.v, b.v));
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_SSE_H_
