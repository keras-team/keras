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

// simd_wrappers_neon.h: SSE SIMD wrappers

#ifndef GEMMLOWP_INTERNAL_SIMD_WRAPPERS_SSE_H_
#define GEMMLOWP_INTERNAL_SIMD_WRAPPERS_SSE_H_

#include <smmintrin.h>

namespace gemmlowp {

using Int32x4 = __m128i;
using Int16x8 = __m128i;
using Uint8x16 = __m128i;

template <int ScalarCount>
struct RegisterType<std::int32_t, ScalarCount> {
  using Type =
      typename std::conditional<ScalarCount >= 4, Int32x4, std::int32_t>::type;
};

template <int ScalarCount>
struct RegisterType<std::int16_t, ScalarCount> {
  using Type =
      typename std::conditional<ScalarCount >= 8, Int16x8, std::int16_t>::type;
};

template <int ScalarCount>
struct RegisterType<std::uint8_t, ScalarCount> {
  using Type = typename std::conditional<
      ScalarCount >= 16, Uint8x16,
      typename std::conditional<ScalarCount >= 4, std::uint32_t,
                                std::uint8_t>::type>::type;
};

inline Int32x4 LoadInt32x4(const std::int32_t* src) {
  return _mm_loadu_si128(reinterpret_cast<const Int32x4*>(src));
}

inline Int32x4 LoadInt16x8(const std::int16_t* src) {
  return _mm_loadu_si128(reinterpret_cast<const Int16x8*>(src));
}

inline void StoreInt32x4(std::int32_t* dst, Int32x4 value) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
}

inline void StoreInt16x8(std::int16_t* dst, Int16x8 value) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
}

inline Uint8x16 LoadUint8x16(const std::uint8_t* src) {
  return _mm_loadu_si128(reinterpret_cast<const Uint8x16*>(src));
}

inline void StoreUint8x16(std::uint8_t* dst, Uint8x16 value) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), value);
}

template <int Lane>
std::int32_t GetLane(Int32x4 value) {
  return _mm_extract_epi32(value, Lane);
}

template <int Lane>
Int32x4 DupLane(Int32x4 value) {
  return _mm_shuffle_epi32(value, _MM_SHUFFLE(Lane, Lane, Lane, Lane));
}

inline Int32x4 Mul(Int32x4 a, std::int32_t b) {
  return Mul(a, Dup<Int32x4>(b));
}

inline Int32x4 Min(Int32x4 a, Int32x4 b) { return _mm_min_epi32(a, b); }

inline Int32x4 Max(Int32x4 a, Int32x4 b) { return _mm_max_epi32(a, b); }

inline Int32x4 SaturatingRoundingDoublingHighMul(Int32x4 a, std::int32_t b) {
  return SaturatingRoundingDoublingHighMul(a, Dup<Int32x4>(b));
}

template <int Lane>
Int32x4 MulByRhsLane(Int32x4 a, Int32x4 b) {
  return Mul(a, DupLane<Lane>(b));
}

inline void MulAdd(Int32x4 lhs, Int32x4 rhs, Int32x4* acc) {
  *acc = Add(*acc, Mul(lhs, rhs));
}

inline void MulAdd(Int32x4 lhs, std::int32_t rhs, Int32x4* acc) {
  *acc = Add(*acc, Mul(lhs, rhs));
}

template <int Lane>
inline void MulAddByRhsLane(Int32x4 lhs, Int32x4 rhs, Int32x4* acc) {
  *acc = Add(*acc, MulByRhsLane<Lane>(lhs, rhs));
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

#endif  // GEMMLOWP_INTERNAL_SIMD_WRAPPERS_SSE_H_
