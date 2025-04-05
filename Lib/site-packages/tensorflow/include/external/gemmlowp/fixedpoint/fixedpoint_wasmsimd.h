// Copyright 2020 Google Inc. All Rights Reserved.
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

// fixedpoint_wasmsimd.h: optimized WAsm SIMD specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_WASMSIMD_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_WASMSIMD_H_

#include <wasm_simd128.h>

namespace gemmlowp {

// WAsm SIMD intrinsics are not typed: there is a single v128_t vector
// type that does not distinguish between "int32x4" and "int16x8" use
// cases, unlike the NEON equivalents. Because we had initially focused
// on int32x4, we did not pay attention and specialized these fixedpoint
// templates directly for v128_t hardcoding the int32x4 semantics,
// not leaving room for int16x8 semantics. Amending that by adding a separate
// data type, int16x8_v128_t, that wraps v128_t while being a separate
// type.
struct int16x8_v128_t {
  v128_t v;
};

// Keep int16x8_v128_t trivially constructible/destructible and provide
// easily optimized helper function.
inline int16x8_v128_t to_int16x8_v128_t(v128_t w) {
  int16x8_v128_t r;
  r.v = w;
  return r;
}

template <>
struct FixedPointRawTypeTraits<v128_t> {
  typedef std::int32_t ScalarRawType;
  static constexpr int kLanes = 4;
};

template <>
struct FixedPointRawTypeTraits<int16x8_v128_t> {
  typedef std::int16_t ScalarRawType;
  static constexpr int kLanes = 8;
};

template <>
inline v128_t BitAnd(v128_t a, v128_t b) {
  return wasm_v128_and(a, b);
}

template <>
inline int16x8_v128_t BitAnd(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_v128_and(a.v, b.v));
}

template <>
inline v128_t BitOr(v128_t a, v128_t b) {
  return wasm_v128_or(a, b);
}

template <>
inline int16x8_v128_t BitOr(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_v128_or(a.v, b.v));
}

template <>
inline v128_t BitXor(v128_t a, v128_t b) {
  return wasm_v128_xor(a, b);
}

template <>
inline int16x8_v128_t BitXor(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_v128_xor(a.v, b.v));
}

template <>
inline v128_t BitNot(v128_t a) {
  return wasm_v128_not(a);
}

template <>
inline int16x8_v128_t BitNot(int16x8_v128_t a) {
  return to_int16x8_v128_t(wasm_v128_not(a.v));
}

template <>
inline v128_t Add(v128_t a, v128_t b) {
  return wasm_i32x4_add(a, b);
}

template <>
inline int16x8_v128_t Add(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_add(a.v, b.v));
}

template <>
inline v128_t Mul(v128_t a, v128_t b) {
  return wasm_i32x4_mul(a, b);
}

template <>
inline int16x8_v128_t Mul(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_mul(a.v, b.v));
}

template <>
inline v128_t Sub(v128_t a, v128_t b) {
  return wasm_i32x4_sub(a, b);
}

template <>
inline int16x8_v128_t Sub(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_sub(a.v, b.v));
}

template <>
inline v128_t Neg(v128_t a) {
  return wasm_i32x4_neg(a);
}

template <>
inline int16x8_v128_t Neg(int16x8_v128_t a) {
  return to_int16x8_v128_t(wasm_i16x8_neg(a.v));
}

template <>
inline v128_t ShiftLeft(v128_t a, int offset) {
  return wasm_i32x4_shl(a, offset);
}

template <>
inline int16x8_v128_t ShiftLeft(int16x8_v128_t a, int offset) {
  return to_int16x8_v128_t(wasm_i16x8_shl(a.v, offset));
}

template <>
inline v128_t ShiftRight(v128_t a, int offset) {
  return wasm_i32x4_shr(a, offset);
}

template <>
inline int16x8_v128_t ShiftRight(int16x8_v128_t a, int offset) {
  return to_int16x8_v128_t(wasm_i16x8_shr(a.v, offset));
}

template <>
inline v128_t SelectUsingMask(v128_t if_mask, v128_t then_val,
                              v128_t else_val) {
  return wasm_v128_bitselect(then_val, else_val, if_mask);
}

template <>
inline int16x8_v128_t SelectUsingMask(int16x8_v128_t if_mask,
                                      int16x8_v128_t then_val,
                                      int16x8_v128_t else_val) {
  return to_int16x8_v128_t(
      wasm_v128_bitselect(then_val.v, else_val.v, if_mask.v));
}

template <>
inline v128_t MaskIfEqual(v128_t a, v128_t b) {
  return wasm_i32x4_eq(a, b);
}

template <>
inline int16x8_v128_t MaskIfEqual(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_eq(a.v, b.v));
}

template <>
inline v128_t MaskIfNotEqual(v128_t a, v128_t b) {
  return wasm_i32x4_ne(a, b);
}

template <>
inline int16x8_v128_t MaskIfNotEqual(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_ne(a.v, b.v));
}

template <>
inline v128_t MaskIfZero(v128_t a) {
  return MaskIfEqual(a, wasm_i32x4_const(0, 0, 0, 0));
}

template <>
inline int16x8_v128_t MaskIfZero(int16x8_v128_t a) {
  return MaskIfEqual(
      a, to_int16x8_v128_t(wasm_i16x8_const(0, 0, 0, 0, 0, 0, 0, 0)));
}

template <>
inline v128_t MaskIfNonZero(v128_t a) {
  return MaskIfNotEqual(a, wasm_i32x4_const(0, 0, 0, 0));
}

template <>
inline int16x8_v128_t MaskIfNonZero(int16x8_v128_t a) {
  return MaskIfNotEqual(
      a, to_int16x8_v128_t(wasm_i16x8_const(0, 0, 0, 0, 0, 0, 0, 0)));
}

template <>
inline v128_t MaskIfGreaterThan(v128_t a, v128_t b) {
  return wasm_i32x4_gt(a, b);
}

template <>
inline int16x8_v128_t MaskIfGreaterThan(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_gt(a.v, b.v));
}

template <>
inline v128_t MaskIfLessThan(v128_t a, v128_t b) {
  return wasm_i32x4_lt(a, b);
}

template <>
inline int16x8_v128_t MaskIfLessThan(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_lt(a.v, b.v));
}

template <>
inline v128_t MaskIfGreaterThanOrEqual(v128_t a, v128_t b) {
  return wasm_i32x4_ge(a, b);
}

template <>
inline int16x8_v128_t MaskIfGreaterThanOrEqual(int16x8_v128_t a,
                                               int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_ge(a.v, b.v));
}

template <>
inline v128_t MaskIfLessThanOrEqual(v128_t a, v128_t b) {
  return wasm_i32x4_le(a, b);
}

template <>
inline int16x8_v128_t MaskIfLessThanOrEqual(int16x8_v128_t a,
                                            int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_le(a.v, b.v));
}

/* Assumptions:
   - All and Any are used on masks.
   - masks are all_ones for true lanes, all_zeroes otherwise.
Hence, All means all 128bits set, and Any means any bit set.
*/

template <>
inline bool All(v128_t a) {
  return wasm_i32x4_all_true(a);
}

template <>
inline bool All(int16x8_v128_t a) {
  return wasm_i16x8_all_true(a.v);
}

template <>
inline bool Any(v128_t a) {
  return wasm_i32x4_any_true(a);
}

template <>
inline bool Any(int16x8_v128_t a) {
  return wasm_i16x8_any_true(a.v);
}

template <>
inline v128_t RoundingHalfSum(v128_t a, v128_t b) {
  // We divide the inputs before the add to avoid the overflow and costly test.
  const v128_t one = wasm_i32x4_const(1, 1, 1, 1);
  const v128_t sign_bit_mask =
      wasm_i32x4_const(0x80000000, 0x80000000, 0x80000000, 0x80000000);
  const v128_t sum = Add(a, b);
  const v128_t rounded_half_sum = ShiftRight(Add(sum, one), 1);
  const v128_t overflow =
      BitAnd(BitAnd(BitXor(a, rounded_half_sum), BitXor(b, rounded_half_sum)),
             sign_bit_mask);
  const v128_t result = BitXor(rounded_half_sum, overflow);
  return result;
}

template <>
inline int16x8_v128_t RoundingHalfSum(int16x8_v128_t a, int16x8_v128_t b) {
  // Idea: go to unsigned to use wasm_u16x8_avgr,
  // borrowed from Intel's arm_neon_sse.h header.
  const v128_t constant_neg_32768 = wasm_i16x8_const(
      -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768);
  const v128_t a_unsigned = wasm_v128_xor(a.v, constant_neg_32768);
  const v128_t b_unsigned = wasm_v128_xor(b.v, constant_neg_32768);
  const v128_t avg_unsigned = wasm_u16x8_avgr(a_unsigned, b_unsigned);
  const v128_t avg = wasm_v128_xor(avg_unsigned, constant_neg_32768);
  return to_int16x8_v128_t(avg);
}

template <>
inline v128_t SaturatingRoundingDoublingHighMul(v128_t a, v128_t b) {
  // TODO: switch to extended multiplication once implemented in the toolchain
  const v128_t a_sign = wasm_i32x4_shr(a, 31);
  const v128_t b_sign = wasm_i32x4_shr(b, 31);

  const v128_t a_ext_lo = wasm_v32x4_shuffle(a, a_sign, 0, 4, 1, 5);
  const v128_t a_ext_hi = wasm_v32x4_shuffle(a, a_sign, 2, 6, 3, 7);
  const v128_t b_ext_lo = wasm_v32x4_shuffle(b, b_sign, 0, 4, 1, 5);
  const v128_t b_ext_hi = wasm_v32x4_shuffle(b, b_sign, 2, 6, 3, 7);

  const v128_t ab_lo = wasm_i64x2_mul(a_ext_lo, b_ext_lo);
  const v128_t ab_hi = wasm_i64x2_mul(a_ext_hi, b_ext_hi);

  const v128_t nudge_2x =
      wasm_i64x2_const(INT64_C(0x80000000), INT64_C(0x80000000));
  const v128_t ab_lo_2x = wasm_i64x2_add(ab_lo, ab_lo);
  const v128_t ab_hi_2x = wasm_i64x2_add(ab_hi, ab_hi);

  const v128_t ab_lo_rounded_2x = wasm_i64x2_add(ab_lo_2x, nudge_2x);
  const v128_t ab_hi_rounded_2x = wasm_i64x2_add(ab_hi_2x, nudge_2x);

  const v128_t prod =
      wasm_v32x4_shuffle(ab_lo_rounded_2x, ab_hi_rounded_2x, 1, 3, 5, 7);

  // Saturation only happen if a == b == INT_MIN, and this is the only case
  // where prod == INT_MIN (0x80000000) instead of INT_MAX (0x7FFFFFFF).
  const v128_t min = wasm_i32x4_const(INT32_C(0x80000000), INT32_C(0x80000000),
                                      INT32_C(0x80000000), INT32_C(0x80000000));

  return wasm_v128_xor(prod, wasm_i32x4_eq(prod, min));
}

template <>
inline int16x8_v128_t SaturatingRoundingDoublingHighMul(int16x8_v128_t a,
                                                        int16x8_v128_t b) {
#if 0
  // TODO: enable if https://github.com/WebAssembly/simd/pull/365 is accepted
  return to_int16x8_v128_t(__builtin_wasm_q15mulr_saturate_s_i16x8(a.v, b.v));
#else
  // TODO: switch to extended multiplication once implemented in the toolchain
  v128_t lo = wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(a.v),
                             wasm_i32x4_widen_low_i16x8(b.v));
  v128_t hi = wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(a.v),
                             wasm_i32x4_widen_high_i16x8(b.v));
  const v128_t inc = wasm_i32x4_const(0x4000, 0x4000, 0x4000, 0x4000);
  lo = wasm_i32x4_add(lo, inc);
  hi = wasm_i32x4_add(hi, inc);
  lo = wasm_i32x4_shr(lo, 15);
  hi = wasm_i32x4_shr(hi, 15);
  return to_int16x8_v128_t(wasm_i16x8_narrow_i32x4(lo, hi));
#endif
}

template <>
inline v128_t Dup<v128_t>(std::int32_t x) {
  return wasm_i32x4_splat(x);
}

template <>
inline int16x8_v128_t Dup<int16x8_v128_t>(std::int16_t x) {
  return to_int16x8_v128_t(wasm_i16x8_splat(x));
}

// So far this is only needed for int16.
template <>
inline int16x8_v128_t SaturatingAdd(int16x8_v128_t a, int16x8_v128_t b) {
  return to_int16x8_v128_t(wasm_i16x8_add_saturate(a.v, b.v));
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_WASMSIMD_H_
