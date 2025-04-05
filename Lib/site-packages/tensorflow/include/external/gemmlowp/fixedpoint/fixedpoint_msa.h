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

// fixedpoint_msa.h: optimized MSA specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_MSA_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_MSA_H_

#include <msa.h>

namespace gemmlowp {

template <>
struct FixedPointRawTypeTraits<v4i32> {
  typedef std::int32_t ScalarRawType;
  static constexpr int kLanes = 4;
};

template <>
struct FixedPointRawTypeTraits<v8i16> {
  typedef std::int16_t ScalarRawType;
  static constexpr int kLanes = 8;
};

template <>
inline v4i32 BitAnd(v4i32 a, v4i32 b) {
  return reinterpret_cast<v4i32>(__builtin_msa_and_v(reinterpret_cast<v16u8>(a),
                                                     reinterpret_cast<v16u8>(b)));
}

template <>
inline v8i16 BitAnd(v8i16 a, v8i16 b) {
  return reinterpret_cast<v8i16>(__builtin_msa_and_v(reinterpret_cast<v16u8>(a),
                                                     reinterpret_cast<v16u8>(b)));
}

template <>
inline v4i32 BitOr(v4i32 a, v4i32 b) {
  return reinterpret_cast<v4i32>(__builtin_msa_or_v(reinterpret_cast<v16u8>(a),
                                                    reinterpret_cast<v16u8>(b)));
}

template <>
inline v8i16 BitOr(v8i16 a, v8i16 b) {
  return reinterpret_cast<v8i16>(__builtin_msa_or_v(reinterpret_cast<v16u8>(a),
                                                    reinterpret_cast<v16u8>(b)));
}

template <>
inline v4i32 BitXor(v4i32 a, v4i32 b) {
  return reinterpret_cast<v4i32>(__builtin_msa_xor_v(reinterpret_cast<v16u8>(a),
                                                     reinterpret_cast<v16u8>(b)));
}

template <>
inline v8i16 BitXor(v8i16 a, v8i16 b) {
  return reinterpret_cast<v8i16>(__builtin_msa_xor_v(reinterpret_cast<v16u8>(a),
                                                     reinterpret_cast<v16u8>(b)));
}

template <>
inline v4i32 BitNot(v4i32 a) {
  return reinterpret_cast<v4i32>(__builtin_msa_nor_v(reinterpret_cast<v16u8>(a),
                                                     reinterpret_cast<v16u8>(a)));
}

template <>
inline v8i16 BitNot(v8i16 a) {
  return reinterpret_cast<v8i16>(__builtin_msa_nor_v(reinterpret_cast<v16u8>(a),
                                                     reinterpret_cast<v16u8>(a)));
}

template <>
inline v4i32 Add(v4i32 a, v4i32 b) {
  return __builtin_msa_addv_w(a, b);
}

template <>
inline v8i16 Add(v8i16 a, v8i16 b) {
  return __builtin_msa_addv_h(a, b);
}

template <>
inline v4i32 Sub(v4i32 a, v4i32 b) {
  return __builtin_msa_subv_w(a, b);
}

template <>
inline v8i16 Sub(v8i16 a, v8i16 b) {
  return __builtin_msa_subv_h(a, b);
}

template <>
inline v4i32 Neg(v4i32 a) {
  v4i32 zeroes = __builtin_msa_ldi_w(0);
  return __builtin_msa_subv_w(zeroes, a);
}

template <>
inline v8i16 Neg(v8i16 a) {
  v8i16 zeroes = __builtin_msa_ldi_h(0);
  return __builtin_msa_subv_h(zeroes, a);
}

template <>
inline v4i32 ShiftLeft(v4i32 a, int offset) {
  return __builtin_msa_sll_w(a, __builtin_msa_fill_w(offset));
}

template <>
inline v8i16 ShiftLeft(v8i16 a, int offset) {
  return __builtin_msa_sll_h(a, __builtin_msa_fill_h(offset));
}

template <>
inline v4i32 ShiftRight(v4i32 a, int offset) {
  return __builtin_msa_sra_w(a, __builtin_msa_fill_w(offset));
}

template <>
inline v8i16 ShiftRight(v8i16 a, int offset) {
  return __builtin_msa_sra_h(a, __builtin_msa_fill_h(offset));
}

template <>
inline v4i32 SelectUsingMask(v4i32 if_mask, v4i32 then_val, v4i32 else_val) {
  if_mask = reinterpret_cast<v4i32>(__builtin_msa_bsel_v(reinterpret_cast<v16u8>(if_mask),
                                                         reinterpret_cast<v16u8>(else_val),
                                                         reinterpret_cast<v16u8>(then_val)));
  return if_mask;
}

template <>
inline v8i16 SelectUsingMask(v8i16 if_mask, v8i16 then_val, v8i16 else_val) {
  if_mask = reinterpret_cast<v8i16>(__builtin_msa_bsel_v(reinterpret_cast<v16u8>(if_mask),
                                                         reinterpret_cast<v16u8>(else_val),
                                                         reinterpret_cast<v16u8>(then_val)));
  return if_mask;
}

template <>
inline v4i32 MaskIfEqual(v4i32 a, v4i32 b) {
  return __builtin_msa_ceq_w(a, b);
}

template <>
inline v8i16 MaskIfEqual(v8i16 a, v8i16 b) {
  return __builtin_msa_ceq_h(a, b);
}

template <>
inline v4i32 MaskIfNotEqual(v4i32 a, v4i32 b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline v8i16 MaskIfNotEqual(v8i16 a, v8i16 b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline v4i32 MaskIfZero(v4i32 a) {
  return __builtin_msa_ceqi_w(a, 0);
}

template <>
inline v8i16 MaskIfZero(v8i16 a) {
  return __builtin_msa_ceqi_h(a, 0);
}

template <>
inline v4i32 MaskIfNonZero(v4i32 a) {
  return BitNot(MaskIfZero(a));
}

template <>
inline v8i16 MaskIfNonZero(v8i16 a) {
  return BitNot(MaskIfZero(a));
}

template <>
inline v4i32 MaskIfGreaterThan(v4i32 a, v4i32 b) {
  return __builtin_msa_clt_s_w(b, a);
}

template <>
inline v8i16 MaskIfGreaterThan(v8i16 a, v8i16 b) {
  return __builtin_msa_clt_s_h(b, a);
}

template <>
inline v4i32 MaskIfGreaterThanOrEqual(v4i32 a, v4i32 b) {
  return __builtin_msa_cle_s_w(b, a);
}

template <>
inline v8i16 MaskIfGreaterThanOrEqual(v8i16 a, v8i16 b) {
  return __builtin_msa_cle_s_h(b, a);
}

template <>
inline v4i32 MaskIfLessThan(v4i32 a, v4i32 b) {
  return __builtin_msa_clt_s_w(a, b);
}

template <>
inline v8i16 MaskIfLessThan(v8i16 a, v8i16 b) {
  return __builtin_msa_clt_s_h(a, b);
}

template <>
inline v4i32 MaskIfLessThanOrEqual(v4i32 a, v4i32 b) {
  return __builtin_msa_cle_s_w(a, b);
}

template <>
inline v8i16 MaskIfLessThanOrEqual(v8i16 a, v8i16 b) {
  return __builtin_msa_cle_s_h(a, b);
}

template <>
inline bool All(v4i32 a) {
  return __builtin_msa_bz_v(reinterpret_cast<v16u8>(BitNot(a)));
}

template <>
inline bool All(v8i16 a) {
  return __builtin_msa_bz_v(reinterpret_cast<v16u8>(BitNot(a)));
}

template <>
inline bool Any(v4i32 a) {
  return __builtin_msa_bnz_v(reinterpret_cast<v16u8>(a));
}

template <>
inline bool Any(v8i16 a) {
  return __builtin_msa_bnz_v(reinterpret_cast<v16u8>(a));
}

template <>
inline v4i32 RoundingHalfSum(v4i32 a, v4i32 b) {
  return __builtin_msa_aver_s_w(a, b);
}

template <>
inline v8i16 RoundingHalfSum(v8i16 a, v8i16 b) {
  return __builtin_msa_aver_s_h(a, b);
}

template <>
inline v4i32 SaturatingRoundingDoublingHighMul(v4i32 a, v4i32 b) {
  return __builtin_msa_mulr_q_w(a, b);
}

template <>
inline v8i16 SaturatingRoundingDoublingHighMul(v8i16 a, v8i16 b) {
  return __builtin_msa_mulr_q_h(a, b);
}

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, v4i32, 1> {
  static v4i32 eval(v4i32 x) {
    static_assert(Exponent >= 0 && Exponent < 32, "");
    if (Exponent < 5) {
      for (int i = 0; i < Exponent; i++) {
        x = __builtin_msa_adds_s_w(x, x);
      }
      return x;
    } else {
      // Saturate each signed 32-bit element to (32 - Exponent)
      // bits (this takes full care of negative elements).
      v4i32 res = __builtin_msa_sat_s_w(x, 31 - Exponent);
      // Set tmp to 0x7FFFFFFF for those elements which staturated
      // to smaller (positive) values and 0 for all others.
      v4i32 tmp = __builtin_msa_srli_w(__builtin_msa_clt_s_w(res, x), 1);
      // Shift the saturated elements. The positive saturated elements
      // will have Exponent trailing zero bits after the shift. Those
      // need to be ones, not zeroes.
      res = __builtin_msa_slli_w(res, Exponent);
      // Finally, set those trailing zero bits to ones.
      res = reinterpret_cast<v4i32>(__builtin_msa_or_v(reinterpret_cast<v16u8>(res),
                                                       reinterpret_cast<v16u8>(tmp)));
      return res;
    }
  }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, v8i16, 1> {
  static v8i16 eval(v8i16 x) {
    static_assert(Exponent >= 0 && Exponent < 16, "");
    if (Exponent < 5) {
      for (int i = 0; i < Exponent; i++) {
        x = __builtin_msa_adds_s_h(x, x);
      }
      return x;
    } else {
      // Saturate each signed 16-bit element to (16 - Exponent)
      // bits (this takes full care of negative elements).
      v8i16 res = __builtin_msa_sat_s_h(x, 15 - Exponent);
      // Set tmp to 0x7FFF for those elements which staturated
      // to smaller (positive) values and 0 for all others.
      v8i16 tmp = __builtin_msa_srli_h(__builtin_msa_clt_s_h(res, x), 1);
      // Shift the saturated elements. The positive saturated elements
      // will have Exponent trailing zero bits after the shift. Those
      // need to be ones, not zeroes.
      res = __builtin_msa_slli_h(res, Exponent);
      // Finally, set those trailing zero bits to ones.
      res = reinterpret_cast<v8i16>(__builtin_msa_or_v(reinterpret_cast<v16u8>(res),
                                                       reinterpret_cast<v16u8>(tmp)));
      return res;
    }
  }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, v4i32, -1> {
  static v4i32 eval(v4i32 x) {
    static_assert(-31 <= Exponent && Exponent <= -1, "");
    // Isolate the sign bits.
    v4i32 sign = __builtin_msa_srli_w(x, 31);
    // Decrement the negative elements by 1 (with saturation).
    x = __builtin_msa_subs_s_w(x, sign);
    // Arithmetic shift right with rounding.
    // The srari instruction rounds all midpoint values towards +infinity.
    // It will correctly round negative midpoint values as we just
    // decremented the negative values by 1.
    return __builtin_msa_srari_w(x, -Exponent);
  }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, v8i16, -1> {
  static v8i16 eval(v8i16 x) {
    static_assert(-15 <= Exponent && Exponent <= -1, "");
    // Isolate the sign bits.
    v8i16 sign = __builtin_msa_srli_h(x, 15);
    // Decrement the negative elements by 1 (with saturation).
    x = __builtin_msa_subs_s_h(x, sign);
    // Arithmetic shift right with rounding.
    // The srari instruction rounds all midpoint values towards +infinity.
    // It will correctly round negative midpoint values as we just
    // decremented the negative values by 1.
    return __builtin_msa_srari_h(x, -Exponent);
  }
};

template <>
inline v4i32 RoundingDivideByPOT(v4i32 x, int exponent) {
  v4i32 e = __builtin_msa_fill_w(exponent);
  // Isolate the sign bits.
  v4i32 sign = __builtin_msa_srli_w(x, 31);
  // Reset them to 0 if exponent is 0.
  sign = __builtin_msa_min_s_w(sign, e);
  // Decrement the negative elements by 1 (with saturation)
  // if exponent is non-zero.
  x = __builtin_msa_subs_s_w(x, sign);
  // Arithmetic shift right with rounding.
  // The srar instruction rounds all midpoint values towards +infinity.
  // It will correctly round negative midpoint values as we just
  // decremented the negative values by 1.
  return __builtin_msa_srar_w(x, e);
}

template <>
inline v8i16 RoundingDivideByPOT(v8i16 x, int exponent) {
  v8i16 e = __builtin_msa_fill_h(exponent);
  // Isolate the sign bits.
  v8i16 sign = __builtin_msa_srli_h(x, 15);
  // Reset them to 0 if exponent is 0.
  sign = __builtin_msa_min_s_h(sign, e);
  // Decrement the negative elements by 1 (with saturation)
  // if exponent is non-zero.
  x = __builtin_msa_subs_s_h(x, sign);
  // Arithmetic shift right with rounding.
  // The srar instruction rounds all midpoint values towards +infinity.
  // It will correctly round negative midpoint values as we just
  // decremented the negative values by 1.
  return __builtin_msa_srar_h(x, e);
}

template <>
inline v4i32 Dup<v4i32>(std::int32_t x) {
  return __builtin_msa_fill_w(x);
}

template <>
inline v8i16 Dup<v8i16>(std::int16_t x) {
  return __builtin_msa_fill_h(x);
}

// So far this is only needed for int16.
template <>
inline v8i16 SaturatingAdd(v8i16 a, v8i16 b) {
  return __builtin_msa_adds_s_h(a, b);
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_MSA_H_
