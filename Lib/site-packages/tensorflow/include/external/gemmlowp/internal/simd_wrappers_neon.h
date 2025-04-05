// Copyright 2017 The Gemmlowp Authors. All Rights Reserved.
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

// simd_wrappers_neon.h: NEON specialization of simd_wrappers.h

#ifndef GEMMLOWP_INTERNAL_SIMD_WRAPPERS_NEON_H_
#define GEMMLOWP_INTERNAL_SIMD_WRAPPERS_NEON_H_

#include <arm_neon.h>

namespace gemmlowp {

using Int32x4 = int32x4_t;
using Int16x4 = int16x4_t;
using Int16x8 = int16x8_t;
using Uint8x8 = uint8x8_t;
using Int8x8 = int8x8_t;

template <int ScalarCount>
struct RegisterType<std::int32_t, ScalarCount> {
  using Type =
      typename std::conditional<ScalarCount >= 4, Int32x4, std::int32_t>::type;
};

template <int ScalarCount>
struct RegisterType<std::int16_t, ScalarCount> {
  using Type = typename std::conditional<
      ScalarCount >= 8, Int16x8,
      typename std::conditional<ScalarCount >= 4, Int16x4,
                                std::int16_t>::type>::type;
};

template <int ScalarCount>
struct RegisterType<std::uint8_t, ScalarCount> {
  using Type = typename std::conditional<
      ScalarCount >= 8, Uint8x8,
      typename std::conditional<ScalarCount >= 4, std::uint32_t,
                                std::uint8_t>::type>::type;
};

template <int ScalarCount>
struct RegisterType<std::int8_t, ScalarCount> {
  using Type = typename std::conditional<
      ScalarCount >= 8, Int8x8,
      typename std::conditional<ScalarCount >= 4, std::int32_t,
                                std::int8_t>::type>::type;
};

inline Int32x4 LoadInt32x4(const std::int32_t* src) { return vld1q_s32(src); }
inline Int16x4 LoadInt16x4(const std::int16_t* src) { return vld1_s16(src); }
inline Int16x8 LoadInt16x8(const std::int16_t* src) { return vld1q_s16(src); }

inline void StoreInt32x4(std::int32_t* dst, Int32x4 value) {
  vst1q_s32(dst, value);
}

inline void StoreInt16x4(std::int16_t* dst, Int16x4 value) {
  vst1_s16(dst, value);
}

inline void StoreInt16x8(std::int16_t* dst, Int16x8 value) {
  vst1q_s16(dst, value);
}

template <int Lane>
std::int32_t GetLane(Int32x4 value) {
  return vgetq_lane_s32(value, Lane);
}

template <int Lane>
Int32x4 DupLane(Int32x4 value) {
  switch (Lane) {
    case 0:
      return vdupq_lane_s32(vget_low_s32(value), 0);
    case 1:
      return vdupq_lane_s32(vget_low_s32(value), 1);
    case 2:
      return vdupq_lane_s32(vget_high_s32(value), 0);
    case 3:
      return vdupq_lane_s32(vget_high_s32(value), 1);
    default:
      static_assert(Lane >= 0 && Lane <= 3, "");
      return vdupq_n_s32(0);
  }
}

inline Int32x4 Mul(Int32x4 a, std::int32_t b) { return vmulq_n_s32(a, b); }

inline Int32x4 Min(Int32x4 a, Int32x4 b) { return vminq_s32(a, b); }

inline Int32x4 Max(Int32x4 a, Int32x4 b) { return vmaxq_s32(a, b); }

inline Int32x4 Max(Int32x4 a, std::int32_t b) {
  return vmaxq_s32(a, vdupq_n_s32(b));
}

inline Int32x4 SaturatingRoundingDoublingHighMul(Int32x4 a, std::int32_t b) {
  return vqrdmulhq_n_s32(a, b);
}

template <int Lane>
Int32x4 MulByRhsLane(Int32x4 a, Int32x4 b) {
  switch (Lane) {
    case 0:
      return vmulq_lane_s32(a, vget_low_s32(b), 0);
    case 1:
      return vmulq_lane_s32(a, vget_low_s32(b), 1);
    case 2:
      return vmulq_lane_s32(a, vget_high_s32(b), 0);
    case 3:
      return vmulq_lane_s32(a, vget_high_s32(b), 1);
    default:
      static_assert(Lane >= 0 && Lane <= 3, "");
      return vdupq_n_s32(0);
  }
}

inline void MulAdd(Int32x4 lhs, Int32x4 rhs, Int32x4* acc) {
  *acc = vmlaq_s32(*acc, lhs, rhs);
}

inline void MulAdd(Int32x4 lhs, std::int32_t rhs, Int32x4* acc) {
  *acc = vmlaq_n_s32(*acc, lhs, rhs);
}

template <int Lane>
inline void MulAddByRhsLane(Int32x4 lhs, Int32x4 rhs, Int32x4* acc) {
  switch (Lane) {
    case 0:
      *acc = vmlaq_lane_s32(*acc, lhs, vget_low_s32(rhs), 0);
      break;
    case 1:
      *acc = vmlaq_lane_s32(*acc, lhs, vget_low_s32(rhs), 1);
      break;
    case 2:
      *acc = vmlaq_lane_s32(*acc, lhs, vget_high_s32(rhs), 0);
      break;
    case 3:
      *acc = vmlaq_lane_s32(*acc, lhs, vget_high_s32(rhs), 1);
      break;
    default:
      static_assert(Lane >= 0 && Lane <= 3, "");
  }
}

template <>
struct LoadContiguousImpl<RegBlockInt16<8, 8>> {
  static RegBlockInt16<8, 8> Run(const std::int16_t* src) {
    RegBlockInt16<8, 8> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = vld1q_s16(src + 8 * i);
    }
    return result;
  }
};

template <>
struct LoadContiguousImpl<RegBlockUint8<8, 8>> {
  static RegBlockUint8<8, 8> Run(const std::uint8_t* src) {
    RegBlockUint8<8, 8> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = vld1_u8(src + 8 * i);
    }
    return result;
  }
};

template <>
struct LoadContiguousImpl<RegBlockInt8<8, 8>> {
  static RegBlockInt8<8, 8> Run(const std::int8_t* src) {
    RegBlockInt8<8, 8> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = vld1_s8(src + 8 * i);
    }
    return result;
  }
};

template <>
struct LoadContiguousImpl<RegBlockInt32<8, 8>> {
  static RegBlockInt32<8, 8> Run(const std::int32_t* src) {
    RegBlockInt32<8, 8> result;
    for (int i = 0; i < 16; i++) {
      result.buf.reg[i] = vld1q_s32(src + 4 * i);
    }
    return result;
  }
};

// 4x1 := 4x1 + 1x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<4, 1>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 + 1x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 4x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<4, 1>, RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 + 1x4
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 + 1x4
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<4, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = ShiftLeft(lhs.buf.reg[1], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[2] = ShiftLeft(lhs.buf.reg[2], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[3] = ShiftLeft(lhs.buf.reg[3], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x4 := 4x4 + 4x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<4, 4>, RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = ShiftLeft(lhs.buf.reg[1], rhs.buf.reg[0]);
    result.buf.reg[2] = ShiftLeft(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = ShiftLeft(lhs.buf.reg[3], rhs.buf.reg[0]);
    return result;
  }
};

// 8x1 := 8x1 + 1x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<8, 1>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    const Int32x4 p = Dup<Int32x4>(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = ShiftLeft(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 + 8x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<8, 1>, RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = ShiftLeft(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 + 1x4
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<8, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = ShiftLeft(lhs.buf.reg[1], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[2] = ShiftLeft(lhs.buf.reg[2], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[3] = ShiftLeft(lhs.buf.reg[3], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[4] = ShiftLeft(lhs.buf.reg[4], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[5] = ShiftLeft(lhs.buf.reg[5], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[6] = ShiftLeft(lhs.buf.reg[6], DupLane<3>(rhs.buf.reg[0]));
    result.buf.reg[7] = ShiftLeft(lhs.buf.reg[7], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 8x4 := 8x4 + 8x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<8, 4>, RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = ShiftLeft(lhs.buf.reg[1], rhs.buf.reg[1]);
    result.buf.reg[2] = ShiftLeft(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = ShiftLeft(lhs.buf.reg[3], rhs.buf.reg[1]);
    result.buf.reg[4] = ShiftLeft(lhs.buf.reg[4], rhs.buf.reg[0]);
    result.buf.reg[5] = ShiftLeft(lhs.buf.reg[5], rhs.buf.reg[1]);
    result.buf.reg[6] = ShiftLeft(lhs.buf.reg[6], rhs.buf.reg[0]);
    result.buf.reg[7] = ShiftLeft(lhs.buf.reg[7], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x8
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<1, 8>, RegBlockInt32<1, 8>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 8>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = ShiftLeft(lhs.buf.reg[1], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x1
template <>
struct BroadcastShiftLeftImpl<RegBlockInt32<1, 8>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] = ShiftLeft(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    result.buf.reg[1] = ShiftLeft(lhs.buf.reg[1], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 1x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<4, 1>,
                                        RegBlockInt32<1, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] =
        RoundingDivideByPOT(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 + 1x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<1, 4>,
                                        RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] =
        RoundingDivideByPOT(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 4x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<4, 1>,
                                        RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = RoundingDivideByPOT(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 + 1x4
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<1, 4>,
                                        RegBlockInt32<1, 4>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = RoundingDivideByPOT(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 + 1x4
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<4, 4>,
                                        RegBlockInt32<1, 4>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] =
        RoundingDivideByPOT(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] =
        RoundingDivideByPOT(lhs.buf.reg[1], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[2] =
        RoundingDivideByPOT(lhs.buf.reg[2], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[3] =
        RoundingDivideByPOT(lhs.buf.reg[3], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x4 := 4x4 + 4x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<4, 4>,
                                        RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] = RoundingDivideByPOT(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = RoundingDivideByPOT(lhs.buf.reg[1], rhs.buf.reg[0]);
    result.buf.reg[2] = RoundingDivideByPOT(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = RoundingDivideByPOT(lhs.buf.reg[3], rhs.buf.reg[0]);
    return result;
  }
};

// 8x1 := 8x1 + 1x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<8, 1>,
                                        RegBlockInt32<1, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    const Int32x4 p = Dup<Int32x4>(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = RoundingDivideByPOT(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 + 8x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<8, 1>,
                                        RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = RoundingDivideByPOT(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 + 1x4
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<8, 4>,
                                        RegBlockInt32<1, 4>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] =
        RoundingDivideByPOT(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] =
        RoundingDivideByPOT(lhs.buf.reg[1], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[2] =
        RoundingDivideByPOT(lhs.buf.reg[2], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[3] =
        RoundingDivideByPOT(lhs.buf.reg[3], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[4] =
        RoundingDivideByPOT(lhs.buf.reg[4], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[5] =
        RoundingDivideByPOT(lhs.buf.reg[5], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[6] =
        RoundingDivideByPOT(lhs.buf.reg[6], DupLane<3>(rhs.buf.reg[0]));
    result.buf.reg[7] =
        RoundingDivideByPOT(lhs.buf.reg[7], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 8x4 := 8x4 + 8x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<8, 4>,
                                        RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] = RoundingDivideByPOT(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = RoundingDivideByPOT(lhs.buf.reg[1], rhs.buf.reg[1]);
    result.buf.reg[2] = RoundingDivideByPOT(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = RoundingDivideByPOT(lhs.buf.reg[3], rhs.buf.reg[1]);
    result.buf.reg[4] = RoundingDivideByPOT(lhs.buf.reg[4], rhs.buf.reg[0]);
    result.buf.reg[5] = RoundingDivideByPOT(lhs.buf.reg[5], rhs.buf.reg[1]);
    result.buf.reg[6] = RoundingDivideByPOT(lhs.buf.reg[6], rhs.buf.reg[0]);
    result.buf.reg[7] = RoundingDivideByPOT(lhs.buf.reg[7], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x8
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<1, 8>,
                                        RegBlockInt32<1, 8>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 8>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] = RoundingDivideByPOT(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = RoundingDivideByPOT(lhs.buf.reg[1], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x1
template <>
struct BroadcastRoundingDivideByPOTImpl<RegBlockInt32<1, 8>,
                                        RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] =
        RoundingDivideByPOT(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    result.buf.reg[1] =
        RoundingDivideByPOT(lhs.buf.reg[1], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

}  // end namespace gemmlowp

#include "simd_wrappers_common_neon_sse.h"

#endif  // GEMMLOWP_INTERNAL_SIMD_WRAPPERS_NEON_H_
