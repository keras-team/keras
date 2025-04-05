// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// fixedpoint_neon.h: optimized NEON specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_NEON_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_NEON_H_

#include <arm_neon.h>

namespace gemmlowp {

template <>
struct FixedPointRawTypeTraits<int32x4_t> {
  typedef std::int32_t ScalarRawType;
  static constexpr int kLanes = 4;
};

template <>
struct FixedPointRawTypeTraits<int16x8_t> {
  typedef std::int16_t ScalarRawType;
  static constexpr int kLanes = 8;
};

template <>
inline int32x4_t BitAnd(int32x4_t a, int32x4_t b) {
  return vandq_s32(a, b);
}

template <>
inline int16x8_t BitAnd(int16x8_t a, int16x8_t b) {
  return vandq_s16(a, b);
}

template <>
inline int32x4_t BitOr(int32x4_t a, int32x4_t b) {
  return vorrq_s32(a, b);
}

template <>
inline int16x8_t BitOr(int16x8_t a, int16x8_t b) {
  return vorrq_s16(a, b);
}

template <>
inline int32x4_t BitXor(int32x4_t a, int32x4_t b) {
  return veorq_s32(a, b);
}

template <>
inline int16x8_t BitXor(int16x8_t a, int16x8_t b) {
  return veorq_s16(a, b);
}

template <>
inline int32x4_t BitNot(int32x4_t a) {
  return veorq_s32(a, vdupq_n_s32(-1));
}

template <>
inline int16x8_t BitNot(int16x8_t a) {
  return veorq_s16(a, vdupq_n_s16(-1));
}

template <>
inline int32x4_t Add(int32x4_t a, int32x4_t b) {
  return vaddq_s32(a, b);
}

template <>
inline int16x8_t Add(int16x8_t a, int16x8_t b) {
  return vaddq_s16(a, b);
}

template <>
inline int32x4_t Sub(int32x4_t a, int32x4_t b) {
  return vsubq_s32(a, b);
}

template <>
inline int16x8_t Sub(int16x8_t a, int16x8_t b) {
  return vsubq_s16(a, b);
}

template <>
inline int32x4_t Neg(int32x4_t a) {
  return vnegq_s32(a);
}

template <>
inline int16x8_t Neg(int16x8_t a) {
  return vnegq_s16(a);
}

template <>
inline int32x4_t ShiftLeft(int32x4_t a, int offset) {
  return vshlq_s32(a, vdupq_n_s32(offset));
}

template <>
inline int16x8_t ShiftLeft(int16x8_t a, int offset) {
  return vshlq_s16(a, vdupq_n_s16(offset));
}

template <>
inline int32x4_t ShiftLeft(int32x4_t a, int32x4_t offset) {
  return vshlq_s32(a, offset);
}

template <>
inline int16x8_t ShiftLeft(int16x8_t a, int16x8_t offset) {
  return vshlq_s16(a, offset);
}

template <>
inline int32x4_t ShiftRight(int32x4_t a, int offset) {
  return vshlq_s32(a, vdupq_n_s32(-offset));
}

template <>
inline int16x8_t ShiftRight(int16x8_t a, int offset) {
  return vshlq_s16(a, vdupq_n_s16(-offset));
}

template <>
inline int32x4_t SelectUsingMask(int32x4_t if_mask, int32x4_t then_val,
                                 int32x4_t else_val) {
  return vbslq_s32(vreinterpretq_u32_s32(if_mask), then_val, else_val);
}

template <>
inline int16x8_t SelectUsingMask(int16x8_t if_mask, int16x8_t then_val,
                                 int16x8_t else_val) {
  return vbslq_s16(vreinterpretq_u16_s16(if_mask), then_val, else_val);
}

template <>
inline int32x4_t MaskIfEqual(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vceqq_s32(a, b));
}

template <>
inline int16x8_t MaskIfEqual(int16x8_t a, int16x8_t b) {
  return vreinterpretq_s16_u16(vceqq_s16(a, b));
}

template <>
inline int32x4_t MaskIfNotEqual(int32x4_t a, int32x4_t b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline int16x8_t MaskIfNotEqual(int16x8_t a, int16x8_t b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline int32x4_t MaskIfZero(int32x4_t a) {
  return MaskIfEqual(a, vdupq_n_s32(0));
}

template <>
inline int16x8_t MaskIfZero(int16x8_t a) {
  return MaskIfEqual(a, vdupq_n_s16(0));
}

template <>
inline int32x4_t MaskIfNonZero(int32x4_t a) {
  return vreinterpretq_s32_u32(vtstq_s32(a, a));
}

template <>
inline int16x8_t MaskIfNonZero(int16x8_t a) {
  return vreinterpretq_s16_u16(vtstq_s16(a, a));
}

template <>
inline int32x4_t MaskIfGreaterThan(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcgtq_s32(a, b));
}

template <>
inline int16x8_t MaskIfGreaterThan(int16x8_t a, int16x8_t b) {
  return vreinterpretq_s16_u16(vcgtq_s16(a, b));
}

template <>
inline int32x4_t MaskIfGreaterThanOrEqual(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcgeq_s32(a, b));
}

template <>
inline int16x8_t MaskIfGreaterThanOrEqual(int16x8_t a, int16x8_t b) {
  return vreinterpretq_s16_u16(vcgeq_s16(a, b));
}

template <>
inline int32x4_t MaskIfLessThan(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcltq_s32(a, b));
}

template <>
inline int16x8_t MaskIfLessThan(int16x8_t a, int16x8_t b) {
  return vreinterpretq_s16_u16(vcltq_s16(a, b));
}

template <>
inline int32x4_t MaskIfLessThanOrEqual(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcleq_s32(a, b));
}

template <>
inline int16x8_t MaskIfLessThanOrEqual(int16x8_t a, int16x8_t b) {
  return vreinterpretq_s16_u16(vcleq_s16(a, b));
}

template <>
inline bool All(int32x4_t a) {
  a = vandq_s32(a, vextq_s32(a, a, 1));
  a = vandq_s32(a, vextq_s32(a, a, 2));
  return vgetq_lane_s32(a, 0);
}

template <>
inline bool All(int16x8_t a) {
  a = vandq_s16(a, vextq_s16(a, a, 1));
  a = vandq_s16(a, vextq_s16(a, a, 2));
  a = vandq_s16(a, vextq_s16(a, a, 4));
  return vgetq_lane_s16(a, 0);
}

template <>
inline bool Any(int32x4_t a) {
  a = vorrq_s32(a, vextq_s32(a, a, 1));
  a = vorrq_s32(a, vextq_s32(a, a, 2));
  return vgetq_lane_s32(a, 0);
}

template <>
inline bool Any(int16x8_t a) {
  a = vorrq_s16(a, vextq_s16(a, a, 1));
  a = vorrq_s16(a, vextq_s16(a, a, 2));
  a = vorrq_s16(a, vextq_s16(a, a, 4));
  return vgetq_lane_s16(a, 0);
}

template <>
inline int32x4_t RoundingHalfSum(int32x4_t a, int32x4_t b) {
  return vrhaddq_s32(a, b);
}

template <>
inline int16x8_t RoundingHalfSum(int16x8_t a, int16x8_t b) {
  return vrhaddq_s16(a, b);
}

template <>
inline int32x4_t SaturatingRoundingDoublingHighMul(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}

template <>
inline int16x8_t SaturatingRoundingDoublingHighMul(int16x8_t a, int16x8_t b) {
  return vqrdmulhq_s16(a, b);
}

template <>
inline int32x4_t RoundingDivideByPOT(int32x4_t x, int exponent) {
  const int32x4_t shift_vec = vdupq_n_s32(-exponent);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
  const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
  return vrshlq_s32(fixed_up_x, shift_vec);
}

template <>
inline int16x8_t RoundingDivideByPOT(int16x8_t x, int exponent) {
  const int16x8_t shift_vec = vdupq_n_s16(-exponent);
  const int16x8_t fixup = vshrq_n_s16(vandq_s16(x, shift_vec), 15);
  const int16x8_t fixed_up_x = vqaddq_s16(x, fixup);
  return vrshlq_s16(fixed_up_x, shift_vec);
}

template <>
inline int32x4_t RoundingDivideByPOT(int32x4_t x, int32x4_t exponent) {
  const int32x4_t shift_vec = vnegq_s32(exponent);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
  const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
  return vrshlq_s32(fixed_up_x, shift_vec);
}

template <>
inline int16x8_t RoundingDivideByPOT(int16x8_t x, int16x8_t exponent) {
  const int16x8_t shift_vec = vnegq_s16(exponent);
  const int16x8_t fixup = vshrq_n_s16(vandq_s16(x, shift_vec), 15);
  const int16x8_t fixed_up_x = vqaddq_s16(x, fixup);
  return vrshlq_s16(fixed_up_x, shift_vec);
}

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32x4_t, 1> {
  static int32x4_t eval(int32x4_t x) { return vqshlq_n_s32(x, Exponent); }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32x4_t, -1> {
  static int32x4_t eval(int32x4_t x) {
    const int32x4_t fixup = vshrq_n_s32(x, 31);
    const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
    return vrshrq_n_s32(fixed_up_x, -Exponent);
  }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int16x8_t, 1> {
  static int16x8_t eval(int16x8_t x) { return vqshlq_n_s16(x, Exponent); }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int16x8_t, -1> {
  static int16x8_t eval(int16x8_t x) {
    const int16x8_t fixup = vshrq_n_s16(x, 15);
    const int16x8_t fixed_up_x = vqaddq_s16(x, fixup);
    return vrshrq_n_s16(fixed_up_x, -Exponent);
  }
};

template <>
inline int32x4_t Dup<int32x4_t>(std::int32_t x) {
  return vdupq_n_s32(x);
}

template <>
inline int16x8_t Dup<int16x8_t>(std::int16_t x) {
  return vdupq_n_s16(x);
}

// So far this is only needed for int16.
template <>
inline int16x8_t SaturatingAdd(int16x8_t a, int16x8_t b) {
  return vqaddq_s16(a, b);
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_NEON_H_
