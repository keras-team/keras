// Copyright 2018 The Gemmlowp Authors. All Rights Reserved.
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

// simd_wrappers_msa.h: MSA specialization of simd_wrappers.h

#ifndef GEMMLOWP_INTERNAL_SIMD_WRAPPERS_MSA_H_
#define GEMMLOWP_INTERNAL_SIMD_WRAPPERS_MSA_H_

#include <msa.h>

namespace gemmlowp {

using Int32x4 = v4i32;
using Int16x8 = v8i16;
using Uint8x16 = v16i8;

template <int ScalarCount>
struct RegisterType<std::int32_t, ScalarCount> {
  using Type =
      typename std::conditional<ScalarCount >= 4, Int32x4, std::int32_t>::type;
};

template <int ScalarCount>
struct RegisterType<std::int16_t, ScalarCount> {
  using Type = typename std::conditional<ScalarCount >= 8, Int16x8, std::int16_t>::type;
};

template <int ScalarCount>
struct RegisterType<std::uint8_t, ScalarCount> {
  using Type = typename std::conditional<
      ScalarCount >= 16, Uint8x16,
      typename std::conditional<ScalarCount >= 4, std::uint32_t,
                                std::uint8_t>::type>::type;
};

inline Int32x4 LoadInt32x4(const std::int32_t* src) {
  return __builtin_msa_ld_w(const_cast<std::int32_t*>(src), 0);
}

inline Int32x4 LoadInt32x4(const Int32x4* src) {
  return __builtin_msa_ld_w(const_cast<Int32x4*>(src), 0);
}

inline void StoreInt32x4(std::int32_t* dst, Int32x4 value) {
  __builtin_msa_st_w(value, dst, 0);
}

inline void StoreInt32x4(Int32x4* dst, Int32x4 value) {
  __builtin_msa_st_w(value, dst, 0);
}

inline Int16x8 LoadInt16x8(const std::int16_t* src) {
  return __builtin_msa_ld_h(const_cast<std::int16_t*>(src), 0);
}

inline Int16x8 LoadInt16x8(const Int16x8* src) {
  return __builtin_msa_ld_h(const_cast<Int16x8*>(src), 0);
}

inline void StoreInt16x8(std::int16_t* dst, Int16x8 value) { __builtin_msa_st_h(value, dst, 0); }

inline void StoreInt16x8(Int16x8* dst, Int16x8 value) { __builtin_msa_st_h(value, dst, 0); }

inline Uint8x16 LoadUint8x16(const std::uint8_t* src) {
  return __builtin_msa_ld_b(const_cast<std::uint8_t*>(src), 0);
}

inline Uint8x16 LoadUint8x16(const Uint8x16* src) {
  return __builtin_msa_ld_b(const_cast<Uint8x16*>(src), 0);
}

inline void StoreUint8x16(std::uint8_t* dst, Uint8x16 value) {
  __builtin_msa_st_b(value, dst, 0);
}

inline void StoreUint8x16(Uint8x16* dst, Uint8x16 value) {
  __builtin_msa_st_b(value, dst, 0);
}

template <int Lane>
std::int32_t GetLane(Int32x4 value) {
  return __builtin_msa_copy_s_w(value, Lane);
}

template <int Lane>
Int32x4 DupLane(Int32x4 value) {
  static_assert(Lane >= 0 && Lane <= 3, "");
  return __builtin_msa_splati_w(value, Lane);
}

inline Int32x4 Mul(Int32x4 a, std::int32_t b) {
  return __builtin_msa_mulv_w(a, __builtin_msa_fill_w(b));
}

inline Int32x4 Min(Int32x4 a, Int32x4 b) { return __builtin_msa_min_s_w(a, b); }

inline Int32x4 Max(Int32x4 a, Int32x4 b) { return __builtin_msa_max_s_w(a, b); }

inline Int32x4 SaturatingRoundingDoublingHighMul(Int32x4 a, std::int32_t b) {
  return __builtin_msa_mulr_q_w(a, __builtin_msa_fill_w(b));
}

template <int Lane>
Int32x4 MulByRhsLane(Int32x4 a, Int32x4 b) {
  static_assert(Lane >= 0 && Lane <= 3, "");
  return __builtin_msa_mulv_w(a, __builtin_msa_splati_w(b, Lane));
}

static inline v4i32 workaround_msa_maddv_w(v4i32 a, v4i32 b, v4i32 c) {
  // Workaround for incorrect encoding of maddv.df in gcc (a exchanged with c).
#if 0
  return __builtin_msa_maddv_w(a, b, c);
#else
  asm volatile("maddv.w %w[a], %w[b], %w[c]\n"
               // Outputs
               : [a] "+f"(a)
               // Inputs
               : [b] "f"(b), [c] "f"(c));
  return a;
#endif
}

inline void MulAdd(Int32x4 lhs, Int32x4 rhs, Int32x4* acc) {
  Int32x4 tmp = LoadInt32x4(acc);
  tmp = workaround_msa_maddv_w(tmp, lhs, rhs);
  StoreInt32x4(acc, tmp);
}

inline void MulAdd(Int32x4 lhs, std::int32_t rhs, Int32x4* acc) {
  Int32x4 tmp = LoadInt32x4(acc);
  tmp = workaround_msa_maddv_w(tmp, lhs, __builtin_msa_fill_w(rhs));
  StoreInt32x4(acc, tmp);
}

template <int Lane>
inline void MulAddByRhsLane(Int32x4 lhs, Int32x4 rhs, Int32x4* acc) {
  static_assert(Lane >= 0 && Lane <= 3, "");
  Int32x4 tmp = LoadInt32x4(acc);
  tmp = workaround_msa_maddv_w(tmp, lhs, __builtin_msa_splati_w(rhs, Lane));
  StoreInt32x4(acc, tmp);
}

template <>
struct LoadContiguousImpl<RegBlockUint8<8, 8>> {
  static RegBlockUint8<8, 8> Run(const std::uint8_t* src) {
    RegBlockUint8<8, 8> result;
    for (int i = 0; i < 4; i++) {
      result.buf.reg[i] = LoadUint8x16(src + 16 * i);
    }
    return result;
  }
};

template <>
struct LoadContiguousImpl<RegBlockInt32<8, 8>> {
  static RegBlockInt32<8, 8> Run(const std::int32_t* src) {
    RegBlockInt32<8, 8> result;
    for (int i = 0; i < 16; i++) {
      result.buf.reg[i] = LoadInt32x4(src + 4 * i);
    }
    return result;
  }
};

template <>
struct LoadContiguousImpl<RegBlockInt16<8, 8>> {
  static RegBlockInt16<8, 8> Run(const std::int16_t* src) {
    RegBlockInt16<8, 8> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = LoadInt16x8(src + 8 * i);
    }
    return result;
  }
};

}  // end namespace gemmlowp

#include "simd_wrappers_common_neon_sse.h"

#endif  // GEMMLOWP_INTERNAL_SIMD_WRAPPERS_MSA_H_
